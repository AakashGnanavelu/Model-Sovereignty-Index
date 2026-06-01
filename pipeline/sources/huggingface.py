"""
Fetch model metadata from Hugging Face Hub API (no auth required for public models).
"""
import requests

HF_API_BASE = "https://huggingface.co/api/models"


def fetch_huggingface_model(model_id: str) -> dict | None:
    """
    Fetch public model info from Hugging Face API.
    model_id: e.g. 'swiss-ai/Apertus-70B-2509' or 'meta-llama/Llama-3.2-1B'
    Returns dict with id, author, downloads, likes, tags, pipeline_tag, library_name,
    license, siblings (files), cardData, etc. Returns None on failure.
    """
    url = f"{HF_API_BASE}/{model_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None
