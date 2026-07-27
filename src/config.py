"""Central configuration — all env-overridable (prefix RAG_)."""
import os
from dataclasses import dataclass, field


def _env(key: str, default, cast=str):
    raw = os.getenv(f"RAG_{key}")
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes")
    return cast(raw)


def _default_gen_model():
    default = os.getenv("RAG_GEN_MODEL", "")
    if default:
        return default
    try:
        import torch
        is_cpu = not torch.cuda.is_available()
    except ImportError:
        is_cpu = True
    return "Qwen/Qwen2.5-0.5B-Instruct" if is_cpu else "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class Settings:
    # models
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    gen_model: str = field(default_factory=_default_gen_model)
    nli_model: str = field(default_factory=lambda: _env("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # verifier
    hrr_dim: int = field(default_factory=lambda: _env("HRR_DIM", 2048, int))
    use_structural: bool = field(default_factory=lambda: _env("USE_STRUCTURAL", True, bool))
    use_entailment: bool = field(default_factory=lambda: _env("USE_ENTAILMENT", True, bool))
    w_embedding: float = field(default_factory=lambda: _env("W_EMBEDDING", 0.30, float))
    w_structural: float = field(default_factory=lambda: _env("W_STRUCTURAL", 0.30, float))
    w_entailment: float = field(default_factory=lambda: _env("W_ENTAILMENT", 0.40, float))
    t_correct: float = field(default_factory=lambda: _env("T_CORRECT", 0.62, float))
    t_incorrect: float = field(default_factory=lambda: _env("T_INCORRECT", 0.40, float))

    # pipeline
    top_k: int = field(default_factory=lambda: _env("TOP_K", 5, int))
    min_correct: int = field(default_factory=lambda: _env("MIN_CORRECT", 2, int))
    max_retries: int = field(default_factory=lambda: _env("MAX_RETRIES", 2, int))
    agent_max_steps: int = field(default_factory=lambda: _env("AGENT_MAX_STEPS", 6, int))
    max_new_tokens: int = field(default_factory=lambda: _env("MAX_NEW_TOKENS", 256, int))
    use_hybrid: bool = field(default_factory=lambda: _env("USE_HYBRID", False, bool))
    quantize: bool = field(default_factory=lambda: _env("QUANTIZE", True, bool))

    # infra
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", "data/index"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: _env("LOG_JSON", False, bool))
    request_timeout_s: float = field(default_factory=lambda: _env("REQUEST_TIMEOUT_S", 120.0, float))

    def validate(self) -> "Settings":
        assert abs(self.w_embedding + self.w_structural + self.w_entailment - 1.0) < 1e-6, \
            "fusion weights must sum to 1.0"
        assert 0 <= self.t_incorrect < self.t_correct <= 1, "thresholds must satisfy 0<=t_inc<t_cor<=1"
        assert self.hrr_dim >= 512, "hrr_dim too small for reliable cleanup"
        return self
