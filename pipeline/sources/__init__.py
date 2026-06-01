from .huggingface import fetch_huggingface_model

__all__ = ["fetch_huggingface_model", "fetch_web_evidence"]


def fetch_web_evidence(*args, **kwargs):
    """Lazy import to avoid pulling in ddgs/trafilatura when only using HF."""
    from .web import fetch_web_evidence as _fetch
    return _fetch(*args, **kwargs)
