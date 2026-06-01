import json
import re
import time
import random
from typing import Any
 
# ── optional third-party imports ──────────────────────────────────────────────
try:
    import trafilatura          # type: ignore
except ImportError:
    trafilatura = None          # type: ignore
 
try:
    from ddgs import DDGS       # type: ignore
except ImportError:
    DDGS = None                 # type: ignore
 
from pipeline.sovereignty_score import (
    CATEGORIES,
    CATEGORY_DESCRIPTION,
    CATEGORY_KEYWORDS,
)
from pipeline.ask import ask_publicai    # provided by the caller's environment
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LLM-based relevance judge
# ══════════════════════════════════════════════════════════════════════════════
 
# System prompt sent once per call (kept short to save tokens)
_SYSTEM = (
    "You are a precise relevance classifier for an AI model research pipeline. "
    "You must respond with valid JSON only — no markdown, no explanation outside the JSON."
)
 
_JUDGE_TEMPLATE = """\
You are checking whether a piece of text is useful evidence for researching \
a specific AI model and a specific topic category.
 
MODEL : {model_name}
CATEGORY : {category_name}
CATEGORY DESCRIPTION : {category_description}
 
TEXT:
\"\"\"
{text}
\"\"\"
 
Decide:
1. Does this text contain information about the model (or a close variant / \
the organisation that made it)?
2. Does this text contain information relevant to the category described above?
 
Both must be true for the text to be relevant.
 
Respond with JSON only, exactly this structure:
{{
  "relevant": true or false,
  "model_mentioned": true or false,
  "category_match": true or false,
  "reason": "one concise sentence"
}}"""
 
 
def _llm_judge(
    text: str,
    model_name: str,
    category_name: str,
    user_agent: str,
    api_key: str | None = None,
    char_limit: int = 1500,
) -> dict[str, Any]:
    """
    Ask the LLM whether *text* is relevant evidence for (*model_name*, *category_name*).
 
    Returns a dict with keys: relevant (bool), model_mentioned (bool),
    category_match (bool), reason (str).
    Falls back to relevant=False on any parse/network error so the pipeline
    never crashes — the error is surfaced in 'reason'.
    """
    category_desc = CATEGORY_DESCRIPTION.get(category_name, category_name)
 
    prompt = _JUDGE_TEMPLATE.format(
        model_name=model_name,
        category_name=category_name,
        category_description=category_desc,
        text=text[:char_limit].strip(),
    )
 
    payload = {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    }
 
    try:
        raw = ask_publicai(
            payload=payload,
            user_agent=user_agent,
            api_key=api_key,
        )
        # Strip markdown fences if the model wraps output anyway
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
 
        # Normalise — ensure all expected keys exist
        return {
            "relevant":        bool(result.get("relevant", False)),
            "model_mentioned": bool(result.get("model_mentioned", False)),
            "category_match":  bool(result.get("category_match", False)),
            "reason":          str(result.get("reason", "")),
        }
 
    except (json.JSONDecodeError, RuntimeError, KeyError) as exc:
        return {
            "relevant":        False,
            "model_mentioned": False,
            "category_match":  False,
            "reason":          f"Judge error: {exc}",
        }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Evidence sentence extractor  (kept — still useful for the downstream pipeline)
# ══════════════════════════════════════════════════════════════════════════════
 
def extract_evidence(
    text: str,
    keywords: list[str],
    max_spans: int = 5,
) -> list[str]:
    """
    Return up to *max_spans* sentences from *text* that contain a keyword.
    Used to give the rest of the pipeline a focused excerpt rather than the
    full page dump.
    """
    seen:  set[str]  = set()
    spans: list[str] = []
 
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.strip()
        if not s or s in seen:
            continue
        if any(kw.lower() in s.lower() for kw in keywords):
            seen.add(s)
            spans.append(s)
        if len(spans) >= max_spans:
            break
 
    return spans
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Web helpers
# ══════════════════════════════════════════════════════════════════════════════
 
def _search_web(query: str, top_k: int = 5) -> list[dict]:
    """DuckDuckGo search → list of {title, snippet, url}."""
    if DDGS is None:
        return []
    try:
        time.sleep(random.uniform(1.0, 2.5))
        with DDGS() as ddgs:
            return [
                {
                    "title":   r.get("title", ""),
                    "snippet": r.get("body",  ""),
                    "url":     r.get("href",  ""),
                }
                for r in ddgs.text(query, max_results=top_k)
            ]
    except Exception:
        return []
 
 
def _extract_page_text(url: str) -> str:
    """Scrape *url* and return the main text body (empty string on failure)."""
    if trafilatura is None:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded) or ""
    except Exception:
        return ""
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
 
_RESET  = "\033[0m"
_BLUE   = "\033[94m"
_GREEN  = "\033[92m"
_WHITE  = "\033[97m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
 
_LEVEL_COLOUR: dict[str, str] = {
    "info":    _BLUE,
    "accept":  _GREEN,
    "general": _WHITE,
    "warn":    _YELLOW,
    "error":   _RED,
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════
 
def fetch_web_evidence(
    model_name: str,
    user_agent: str,
    api_key: str | None = None,
    top_k_per_query: int = 5,
    delay_between_requests: float = 1.0,
    scrape_pages: bool = True,
    judge_char_limit: int = 1500,
    verbose: bool = True,
) -> list[dict]:
    """
    Search for evidence about *model_name* across all sovereignty categories
    and return accepted documents.
 
    Parameters
    ----------
    model_name:
        HuggingFace model ID, e.g. "meta-llama/Llama-2-7b".
    user_agent:
        Required by ask_publicai (passed through unchanged).
    api_key:
        PublicAI key. Falls back to PUBLICAI_KEY env var if None.
    top_k_per_query:
        Number of search results to fetch per category.
    delay_between_requests:
        Polite delay (seconds) between full-page scrape requests.
    scrape_pages:
        If False, only Track 1 (snippet judge) runs — faster but shallower.
    judge_char_limit:
        How many characters of page text to send to the LLM judge (Track 2).
        1 500 chars ≈ ~350 tokens, well within the model's context.
    verbose:
        Print progress to stdout.
 
    Returns
    -------
    List of dicts, each with:
        query, url, title, category, track, extracted (evidence text),
        judge (the raw LLM judgment dict)
    """
 
    def _log(msg: str, *, level: str = "info") -> None:
        if not verbose:
            return
        colour = _LEVEL_COLOUR.get(level, _BLUE)
        print(f"{colour}{msg}{_RESET}", flush=True)
 
    if DDGS is None:
        _log("ddgs not installed — run: pip install ddgs", level="error")
        return []
    if trafilatura is None and scrape_pages:
        _log("trafilatura not installed — Track 2 disabled. run: pip install trafilatura", level="warn")
        scrape_pages = False
 
    docs: list[dict] = []
    seen_urls: set[str] = set()   # persists across all categories
 
    for i, category in enumerate(CATEGORIES):
        keywords = CATEGORY_KEYWORDS.get(category, [])
        query    = f"{model_name} {category}"
        _log(f"\n[{i + 1}/{len(CATEGORIES)}] Query: {query!r}")
 
        try:
            results = _search_web(query, top_k=top_k_per_query)
        except Exception as exc:
            _log(f"  Search failed: {exc}", level="error")
            results = []
 
        if not results:
            _log("  No results.", level="warn")
 
        for result in results:
            url     = result.get("url",     "")
            snippet = result.get("snippet", "").strip()
            title   = result.get("title",   "")
 
            if not url:
                continue
 
            if url in seen_urls:
                _log(f"  [skip]    {url[:70]}  (already collected)", level="general")
                continue
 
            accepted      = False
            evidence_text = snippet
            track         = "snippet"
            judgment: dict[str, Any] = {}
 
            # ── Track 1: judge the snippet (no HTTP request) ──────────────────
            if snippet and len(snippet) >= 40:
                judgment = _llm_judge(
                    text=f"{title}\n\n{snippet}",
                    model_name=model_name,
                    category_name=category,
                    user_agent=user_agent,
                    api_key=api_key,
                    char_limit=judge_char_limit,
                )
                _log(
                    f"  [snippet] {url[:70]}\n"
                    f"            → relevant={judgment['relevant']}  "
                    f"model={judgment['model_mentioned']}  "
                    f"cat={judgment['category_match']}\n"
                    f"            reason: {judgment['reason']}",
                    level="general",
                )
                if judgment["relevant"]:
                    accepted      = True
                    evidence_text = snippet
 
            # ── Track 2: scrape + judge full page (only if Track 1 failed) ────
            if not accepted and scrape_pages:
                track     = "page"
                page_text = _extract_page_text(url)
 
                if not page_text or len(page_text.strip()) < 100:
                    _log(f"  [page]    {url[:70]}  (empty / too short)", level="general")
                    time.sleep(delay_between_requests)
                    continue
 
                judgment = _llm_judge(
                    text=page_text,
                    model_name=model_name,
                    category_name=category,
                    user_agent=user_agent,
                    api_key=api_key,
                    char_limit=judge_char_limit,
                )
                _log(
                    f"  [page]    {url[:70]}\n"
                    f"            → relevant={judgment['relevant']}  "
                    f"model={judgment['model_mentioned']}  "
                    f"cat={judgment['category_match']}\n"
                    f"            reason: {judgment['reason']}",
                    level="general",
                )
                if judgment["relevant"]:
                    accepted      = True
                    # Extract focused sentences rather than dumping the full page
                    spans         = extract_evidence(page_text, keywords)
                    evidence_text = " ".join(spans) if spans else page_text[:judge_char_limit]
 
                time.sleep(delay_between_requests)
 
            # ── record ────────────────────────────────────────────────────────
            if accepted:
                seen_urls.add(url)
                docs.append({
                    "query":     query,
                    "url":       url,
                    "title":     title,
                    "category":  category,
                    "track":     track,
                    "extracted": evidence_text,
                    "judge":     judgment,
                })
                _log(f"  ✓ Accepted [{track}] ({len(docs)} total): {url}", level="accept")
            else:
                _log(f"  ✗ Rejected: {url}", level="general")
 
    _log(f"\nDone. {len(docs)} documents collected across {len(CATEGORIES)} categories.")
    return docs