"""
Holographic Reduced Representations (HRR) core operations.

This is the symbolic backbone of the verification layer. We use circular
convolution binding (Plate, 1995) to encode (query_slot, chunk_content)
structures and check whether retrieved chunks are *structurally* consistent
with the query — not just cosine-similar.

Key insight for the paper: embedding similarity conflates topical overlap
with answerability. HRR binding lets us ask "does this chunk contain a
filler for the role the query is asking about?" which is a stricter test.
"""

import numpy as np
from numpy.fft import fft, ifft


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def random_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random HRR vector, i.i.d. N(0, 1/dim) — unit expected norm."""
    return rng.normal(0.0, 1.0 / np.sqrt(dim), size=dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution binding: a ⊛ b."""
    return np.real(ifft(fft(a) * fft(b)))


def unbind(trace: np.ndarray, cue: np.ndarray) -> np.ndarray:
    """Approximate unbinding via correlation: trace ⊛ cue†."""
    return np.real(ifft(fft(trace) * np.conj(fft(cue))))


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Superposition (normalized sum)."""
    return normalize(np.sum(vectors, axis=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


class CleanupMemory:
    """Item memory for cleanup: maps noisy unbound vectors back to the
    nearest known symbol. Essential — raw unbind output is noisy."""

    def __init__(self):
        self.names: list[str] = []
        self.matrix: np.ndarray | None = None

    def add(self, name: str, vector: np.ndarray):
        self.names.append(name)
        v = normalize(vector)[None, :]
        self.matrix = v if self.matrix is None else np.vstack([self.matrix, v])

    def cleanup(self, noisy: np.ndarray, top_k: int = 1):
        """Return [(name, similarity)] of the top_k nearest stored symbols."""
        if self.matrix is None:
            return []
        sims = self.matrix @ normalize(noisy)
        idx = np.argsort(-sims)[:top_k]
        return [(self.names[i], float(sims[i])) for i in idx]


class RoleFillerEncoder:
    """
    Encodes text structures as role-filler HRR traces.

    For a query like "Who directed Inception?":
        trace_q = bind(ROLE_director, FILLER_unknown) + bind(ROLE_subject, FILLER_inception)

    For a chunk "Christopher Nolan directed Inception (2010)...":
        trace_c = bind(ROLE_director, FILLER_nolan) + bind(ROLE_subject, FILLER_inception) + ...

    Verification then unbinds trace_c with the query's open role and checks
    whether the result cleans up to a *concrete* filler (answerable) vs noise
    (topically similar but not answer-bearing).
    """

    def __init__(self, dim: int = 2048, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._symbols: dict[str, np.ndarray] = {}
        self.cleanup_mem = CleanupMemory()

    def symbol(self, name: str) -> np.ndarray:
        """Get-or-create a stable random vector for a symbol name."""
        if name not in self._symbols:
            v = random_vector(self.dim, self.rng)
            self._symbols[name] = v
            self.cleanup_mem.add(name, v)
        return self._symbols[name]

    def encode_structure(self, role_filler_pairs: list[tuple[str, str]]) -> np.ndarray:
        """Encode a list of (role, filler) string pairs into one trace."""
        bound = [
            bind(self.symbol(f"role::{r}"), self.symbol(f"filler::{f}"))
            for r, f in role_filler_pairs
        ]
        return bundle(bound)

    def probe(self, trace: np.ndarray, role: str, top_k: int = 3):
        """Unbind a role from a trace and clean up to candidate fillers."""
        noisy = unbind(trace, self.symbol(f"role::{role}"))
        results = self.cleanup_mem.cleanup(noisy, top_k=top_k)
        return [(n, s) for n, s in results if n.startswith("filler::")]
