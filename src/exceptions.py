"""Typed exceptions so callers can handle failure modes distinctly."""


class RAGError(Exception):
    """Base class for all pipeline errors."""


class ModelLoadError(RAGError):
    """A model or resource (spacy, NLI, generator) failed to load."""


class IndexNotBuiltError(RAGError):
    """Search was called before documents were indexed."""


class GenerationError(RAGError):
    """The generator failed to produce output."""
