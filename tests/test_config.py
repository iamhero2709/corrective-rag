import pytest

from src.config import Settings


def test_defaults_validate():
    Settings().validate()


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_USE_ENTAILMENT", "false")
    s = Settings()
    assert s.top_k == 9
    assert s.use_entailment is False


def test_bad_weights_rejected():
    s = Settings.__new__(Settings)
    object.__setattr__(s, "w_embedding", 0.9)
    object.__setattr__(s, "w_structural", 0.9)
    object.__setattr__(s, "w_entailment", 0.9)
    object.__setattr__(s, "t_correct", 0.6)
    object.__setattr__(s, "t_incorrect", 0.4)
    object.__setattr__(s, "hrr_dim", 2048)
    with pytest.raises(AssertionError):
        s.validate()
