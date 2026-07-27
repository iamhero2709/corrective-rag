"""
Evaluation harness.

Metrics:
  - EM / F1 (standard QA)
  - Abstention accuracy: does the system abstain exactly when it should?
  - Verifier quality: precision/recall of CORRECT verdicts vs gold answer-bearing chunks
  - Hallucination rate proxy: answers rated UNSUPPORTED by an external judge

Datasets to wire in (all on HF hub):
  - hotpot_qa (distractor setting)  -> multi-hop, your main benchmark
  - din0s/asqa                      -> ambiguous long-form QA
  - PopQA                           -> tail knowledge, tests abstention

Ablations for the paper (run each config, log to results/):
  A0: no verification (vanilla RAG)          — baseline
  A1: embedding-only threshold               — cheap baseline
  A2: + NLI entailment                       — CRAG-equivalent
  A3: + HRR structural (full system)         — yours
  A4: HRR-only (no NLI)                      — isolates your contribution
"""

import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(records: list[dict]) -> dict:
    """records: [{"pred": str, "gold": [str], "abstained": bool, "answerable": bool}]"""
    em, f1, n = 0.0, 0.0, 0
    abstain_correct, abstain_total = 0, 0

    for r in records:
        if r.get("answerable", True):
            n += 1
            if r["abstained"]:
                continue  # counts as 0 EM/F1
            em += max(exact_match(r["pred"], g) for g in r["gold"])
            f1 += max(f1_score(r["pred"], g) for g in r["gold"])
        else:
            abstain_total += 1
            abstain_correct += int(r["abstained"])

    out = {
        "n_answerable": n,
        "em": round(em / n, 4) if n else None,
        "f1": round(f1 / n, 4) if n else None,
    }
    if abstain_total:
        out["abstention_acc"] = round(abstain_correct / abstain_total, 4)
    return out


def save_results(config_name: str, metrics: dict, traces: list, out_dir: str = "results"):
    p = Path(out_dir)
    p.mkdir(exist_ok=True)
    (p / f"{config_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (p / f"{config_name}_traces.jsonl").write_text(
        "\n".join(json.dumps(t) for t in traces)
    )
    print(f"[{config_name}] {metrics}")
