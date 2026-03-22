"""
Fetch evidence from the web: search (DuckDuckGo) + scrape (trafilatura).
"""
import time
import random

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


def _search_web(query: str, top_k: int = 5) -> list[dict]:
    """Search DuckDuckGo; return list of {title, snippet, url}."""
    if DDGS is None:
        return []
    try:
        time.sleep(random.uniform(1.0, 2.5))
        with DDGS() as ddgs:
            return [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in ddgs.text(query, max_results=top_k)
            ]
    except Exception:
        return []

# Sovereignty categories used for research (aligned with browser.ipynb)
SOVEREIGNTY_CATEGORIES = [
    "Is the training data private?",
    "Is the model trained locally?",
    "Does the company have full control of the model weights?",
    "Is the model trained on a different model?",
    "Is the model weights private?",
    "Is there country specific knowledge?",
]


def _extract_text(url: str) -> str:
    if not trafilatura:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded) or ""
    except Exception:
        return ""


def fetch_web_evidence(
    model_name: str,
    categories: list[str] | None = None,
    top_k_per_query: int = 5,
    delay_between_requests: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """
    For each sovereignty category, search the web for '{model_name} {category}',
    then scrape each result URL with trafilatura. Returns list of
    { "query", "url", "category", "extracted" }.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    categories = categories or SOVEREIGNTY_CATEGORIES
    docs = []

    if DDGS is None:
        _log("  Install 'ddgs' for web search: pip install ddgs")
        return docs
    if trafilatura is None:
        _log("  Install 'trafilatura' for scraping: pip install trafilatura")

    for i, category in enumerate(categories):
        query = f"{model_name} {category}"
        _log(f"  [{i+1}/{len(categories)}] Searching: {query[:60]}...")
        try:
            results = _search_web(query, top_k=top_k_per_query)
        except Exception as e:
            _log(f"    Search failed: {e}")
            results = []
        for r in results:
            url = r.get("url", "")
            if not url:
                continue
            text = _extract_text(url)
            if not text or len(text.strip()) < 100:
                continue
            docs.append({
                "query": query,
                "url": url,
                "category": category,
                "extracted": text[:8000],
            })
            _log(f"    Scraped ({len(docs)}): {url[:70]}...")
            time.sleep(delay_between_requests)
    return docs
