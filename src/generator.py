from dataclasses import dataclass
from typing import Generator
import logging

log = logging.getLogger("rag.generator")

GEN_PROMPT = """You are a precise assistant. Answer the question using ONLY the provided context. If the context does not contain the answer, say exactly: "INSUFFICIENT_CONTEXT".

Context:
{context}

Question: {question}

Rules:
- Give the SHORT answer only (a name, date, number, yes/no, or brief phrase)
- Do NOT write full sentences or explanations

Answer:"""

CHECK_PROMPT = """You are a strict fact-checker. Given a context, a question, and a proposed answer, classify the answer.

Context:
{context}

Question: {question}
Proposed answer: {answer}

Reply with exactly one word:
- SUPPORTED (every claim in the answer is backed by the context)
- PARTIAL (some claims backed, some not)
- UNSUPPORTED (the answer is not backed by the context)

One word:"""


@dataclass
class GenerationOutput:
    answer: str
    self_check: str
    insufficient: bool


CPU_MODELS = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-1.5B-Instruct",
}


class SmallModelGenerator:
    def __init__(
        self,
        model_name: str = "",
        device: str = "auto",
        max_new_tokens: int = 256,
        quantize: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        is_cpu = device == "cpu"

        if not model_name:
            model_name = CPU_MODELS["small"] if is_cpu else CPU_MODELS["medium"]

        log.info("loading %s on %s (quantize=%s)", model_name, device, quantize)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        def _model_size_ok(mname: str) -> bool:
            """Check if total storage for this model is reasonable (< 2GB).
            1.5B in float16 ≈ 3GB, in float32 ≈ 6GB. Use float16."""
            name_lower = mname.lower()
            for big_marker in ("1.5b", "1b", "2b", "3b", "7b", "llama", "gemma"):
                if big_marker in name_lower:
                    return False
            return True

        kw = {}
        if is_cpu:
            if not _model_size_ok(model_name):
                log.info("large model — using float16 (4-bit not available on CPU)")
                kw["dtype"] = torch.float16
            else:
                kw["dtype"] = torch.float32
            kw["device_map"] = "cpu"
        else:
            kw["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if "device_map" not in kw:
                kw["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if not kw.get("device_map") and device == "cpu":
            self.model = self.model.to("cpu")
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _chat(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        dev = self.device if hasattr(self.model, "device") else "cpu"
        input_ids = inputs["input_ids"].to(dev) if hasattr(inputs, "__getitem__") else inputs.to(dev)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def _chat_stream(self, prompt: str, max_new_tokens: int | None = None) -> Generator[str, None, None]:
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        dev = self.device if hasattr(self.model, "device") else "cpu"
        input_ids = inputs["input_ids"].to(dev) if hasattr(inputs, "__getitem__") else inputs.to(dev)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                         skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        yield from streamer

    def generate_stream(self, question: str, context_chunks: list[str]) -> Generator[str, None, None]:
        context = "\n\n---\n\n".join(context_chunks)
        yield from self._chat_stream(GEN_PROMPT.format(context=context, question=question))

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        return self._chat(GEN_PROMPT.format(context=context, question=question))

    def self_check(self, question: str, context_chunks: list[str], answer: str) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        raw = self._chat(
            CHECK_PROMPT.format(context=context, question=question, answer=answer),
            max_new_tokens=8,
        ).upper()
        for label in ("SUPPORTED", "PARTIAL", "UNSUPPORTED"):
            if label in raw:
                return label
        return "PARTIAL"

    def answer(self, question: str, context_chunks: list[str], with_check: bool = True) -> GenerationOutput:
        ans = self.generate(question, context_chunks)
        insufficient = "INSUFFICIENT_CONTEXT" in ans
        check = "SKIPPED"
        if with_check and not insufficient:
            check = self.self_check(question, context_chunks, ans)
        return GenerationOutput(answer=ans, self_check=check, insufficient=insufficient)
