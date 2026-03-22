"""
Compute a 0-100 sovereignty score from Hugging Face metadata and optional web evidence.
Higher = more sovereign (local control, transparent data, open weights, etc.).
"""
import re
import sys
from pathlib import Path
import os
from pipeline.ask import ask_publicai
import json
from typing import Any, Optional

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Open / permissive licenses (higher sovereignty when weights are open)
OPEN_LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0",
    "cc-by-sa-4.0", "openrail", "openrail++", "bigscience-openrail-m",
    "afl-3.0", "artistic-2.0", "gpl-3.0", "lgpl-3.0", "llama2",
}

# Known public / state-backed orgs (often higher sovereignty)
SOVEREIGNTY_ORGS = {
    "swiss-ai", "ai-sweden", "uk-government", "nvidia", "meta",
    "mistralai", "huggingface", "eleuther.ai", "stabilityai",
}

# Categories and their weight in the overall score (sum = 1.0)
CATEGORIES = [
    "Is the training data private?",
    "Is the model trained locally?",
    "Does the company have full control of the model weights?",
    "Is the model trained on a different model?",
    "Is the model weights private?",
    "Is there country specific knowledge?",
]

DEFAULT_WEIGHTS = {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}


# Organisation-level heuristics
ORG_PUBLIC_HINTS = [
    "university",
    "universität",
    "epfl",
    "eth zurich",
    "cscs",
    "swiss-ai",
    "ai sweden",
    "public ai",
    "government",
    "gov",
    "ministry",
    "federal",
    "national lab",
    "research council",
    "european commission",
    "european union",
    "erc",
]

ORG_BIG_TECH_HINTS = [
    "openai",
    "google",
    "deepmind",
    "alphabet",
    "microsoft",
    "azure",
    "meta",
    "facebook",
    "amazon",
    "aws",
    "anthropic",
    "x.ai",
    "baidu",
    "alibaba",
    "tencent",
    "bytedance",
    "nvidia",
]


def _parse_float(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    m = re.search(r"0?\.\d+|\d+\.?\d*", s.replace(",", "."))
    if m:
        v = float(m.group())
        return max(0.0, min(1.0, v))
    return None


def _normalize_for_quote_match(text: str) -> str:
    """Collapse whitespace for robust substring checks (quotes vs extracted HTML)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _quote_verified_in_sources(quote: str, docs: list[dict], min_overlap: int = 32) -> bool:
    """
    Reject likely hallucinated or paraphrased 'quotes' by requiring a long
    substring of the quote to appear verbatim in extracted source text.
    """
    qn = _normalize_for_quote_match(quote)
    if len(qn) < min_overlap:
        return False
    blobs = [_normalize_for_quote_match((d.get("extracted") or "")) for d in docs]
    # Sliding windows on quote for fuzzy match if the model trimmed start/end
    for step in (0, min(20, max(0, len(qn) // 4))):
        tail = qn[step:]
        if len(tail) < min_overlap:
            break
        for i in range(0, len(tail) - min_overlap + 1, max(1, min_overlap // 2)):
            window = tail[i : i + min_overlap]
            if any(window in blob for blob in blobs):
                return True
    return False


def score_from_huggingface(hf_model: dict) -> dict[str, float]:
    """
    Derive 0-1 component scores from Hugging Face model API response.
    hf_model: from pipeline.sources.huggingface.fetch_huggingface_model()
    """
    scores = {c: 0.5 for c in CATEGORIES}  # default neutral

    if not hf_model:
        return scores

    author = (hf_model.get("author") or "").lower()
    model_id = (hf_model.get("id") or "").lower()
    tags = [t.lower() for t in (hf_model.get("tags") or [])]
    # License can be string or list in API
    raw_license = hf_model.get("license") or ""
    if isinstance(raw_license, list):
        licenses = [str(x).lower() for x in raw_license]
    else:
        licenses = [str(raw_license).lower()] if raw_license else []

    # Is the model weights private? -> open license = more sovereign (weights not private)
    open_license = any(any(ol in lic for ol in OPEN_LICENSES) for lic in licenses)
    if open_license:
        scores["Is the model weights private?"] = 0.2  # low = weights are public
    else:
        scores["Is the model weights private?"] = 0.8  # assume more private

    # Does the company have full control of the model weights? -> open weights = can self-host
    scores["Does the company have full control of the model weights?"] = 0.8 if open_license else 0.3

    # Is the model trained on a different model? -> no base model = from scratch = more sovereign
    card_data = hf_model.get("cardData")
    if not isinstance(card_data, dict):
        card_data = {}
    base_model = (card_data.get("base_model") or hf_model.get("base_model") or "")
    if isinstance(base_model, list):
        if len(base_model) == 0:
            return -1
        base_model = base_model[0].lower()
    elif isinstance(base_model, str):
        base_model=base_model.lower()
    has_base_model = bool(base_model.strip())
    if not has_base_model:
        # Check tags for fine-tune
        if any("finetuned" in t or "fine-tuned" in t or "derived" in t for t in tags):
            scores["Is the model trained on a different model?"] = 0.3
        else:
            scores["Is the model trained on a different model?"] = 0.8  # from scratch
    else:
        scores["Is the model trained on a different model?"] = 0.25

    # Public/state org hint -> more likely local training and country-specific
    org_hint = author in SOVEREIGNTY_ORGS or any(o in model_id for o in SOVEREIGNTY_ORGS)
    if org_hint:
        scores["Is the model trained locally?"] = 0.7
        scores["Is there country specific knowledge?"] = 0.65
    else:
        scores["Is the model trained locally?"] = 0.45
        scores["Is there country specific knowledge?"] = 0.45

    # Training data private: open model card / transparency tags
    if "transparent" in " ".join(tags) or "open" in " ".join(tags) or "fully open" in str(hf_model).lower():
        scores["Is the training data private?"] = 0.3  # data is more open
    else:
        scores["Is the training data private?"] = 0.5

    return scores

def score_from_web_docs(
    web_docs: list[dict],
    model_name: str,
    weights: dict[str, float] | None = None,
    num_of_attempts=2
) -> dict:

    weights = weights or DEFAULT_WEIGHTS

    if not os.getenv("PUBLICAI_KEY"):
        return None

    results = {}

    for i, category in enumerate(CATEGORIES):

        filtered = [
            d for d in web_docs
            if category.lower() in (d.get("category") or "").lower()
        ]

        if not filtered:
            results[category] = {
                "score": 0.5,
                "confidence": 0.0,
                "evidence": []
            }
            continue

        sources = "\n\n".join(
            f"Source {j+1}:\nURL: {d.get('url', '')}\nContent:\n{d.get('extracted', '')[:1200]}"
            for j, d in enumerate(filtered[:5])
        )
     

        prompt = f"""
            You are a strict JSON API.

            Return ONLY valid JSON. No text before or after.

            Schema:
            {{
            "score": number between 0 and 1,
            "confidence": number between 0 and 1,
            "quotes": [
                {{
                "quote": "verbatim copy-paste from the source Content below",
                "url": "exact URL of that source (must match a Source block)",
                "rationale": "one or two sentences: state how this exact wording implies the score you chose for this category (higher = more sovereign for this dimension; say what in the quote pushes the score up or down)"
                }}
            ]
            }}

            Rules:
            - Output MUST be valid JSON
            - No explanations outside the JSON
            - No markdown
            - No trailing commas

            Quote quality (VERY IMPORTANT):
            - Copy the "quote" character-for-character from the Content of one Source block (no paraphrase, no invention)
            - Length: prefer 40–350 characters; never under 25 characters unless the source only has a short sentence
            - Must be a continuous span from the source (not stitched fragments)
            - Exclude boilerplate: navigation, cookie banners, "click here", copyright-only lines, social share text
            - Each quote MUST directly address: "{category}"
            - Reject vague marketing or generic capability claims unless they explicitly touch this category
            - If no passage meets these bars, return "quotes": []

            Scoring (remember: for these yes/no style questions, higher score = MORE sovereign / local control / openness as defined by the category wording):
            - Base score and confidence ONLY on the Sources; do not use outside knowledge
            - "confidence" reflects how clear and on-topic the evidence is (0 = ambiguous or thin, 1 = explicit)
            - Strong, explicit evidence → higher score and higher confidence
            - Weak, indirect, or missing evidence → score near 0.5 and low confidence

            Sources:
            {sources}

            Task:
            For "{model_name}", score the category "{category}" and return up to 3 best quotes with rationale tied to your score.
            """

        try:
            parsed = None
            for _ in range(num_of_attempts):
                content = ask_publicai(
                    prompt=prompt,
                    user_agent="Sovereignty-Pipeline/1.0",
                )

                # print("\n=== RAW LLM OUTPUT ===")
                # print(content)
                # print("=====================\n")

                parsed = extract_valid_json(content)
                # print(f"Parsed content: {parsed}")
                if parsed:
                    break

            if not parsed:
                raise ValueError("Invalid JSON from LLM after retries")

            score = _parse_float(parsed.get("score")) or 0.5
            confidence = _parse_float(parsed.get("confidence"))
            if confidence is None:
                confidence = 0.0
            quotes = parsed.get("quotes", []) or []

            clean_quotes = []
            seen_norm = set()
            for q in quotes:
                text = (q.get("quote") or "").strip()
                url = (q.get("url") or "").strip()
                rationale = (q.get("rationale") or "").strip()

                if len(text) < 25:
                    continue
                if text.lower().startswith("source"):
                    continue
                if not _quote_verified_in_sources(text, filtered):
                    continue
                dedupe_key = _normalize_for_quote_match(text)[:240]
                if dedupe_key in seen_norm:
                    continue
                seen_norm.add(dedupe_key)
                if len(rationale) < 15:
                    rationale = (
                        f"This excerpt was used as supporting text for the category "
                        f"\"{category}\" when estimating the sovereignty-related score."
                    )

                clean_quotes.append({
                    "quote": text,
                    "url": url,
                    "rationale": rationale,
                })

            if not clean_quotes:
                confidence = min(float(confidence), 0.35)

            # Confidence-weighted score
            final_score = score * (0.5 + 0.5 * confidence)

            results[category] = {
                "score": final_score,
                "confidence": confidence,
                "evidence": clean_quotes[:3]
            }

        except Exception as e:
            print(f"Error message: {e}")
            # fallback
            fallback_quotes = []

            for d in filtered[:2]:
                text = (d.get("extracted") or "")[:200].strip()
                if text:
                    fallback_quotes.append({
                        "quote": text,
                        "url": d.get("url", ""),
                        "rationale": (
                            f"Raw excerpt from web evidence for \"{category}\" "
                            f"(automated fallback; LLM scoring failed)."
                        ),
                    })

            results[category] = {
                "score": 0.5,
                "confidence": 0.0,
                "evidence": fallback_quotes
            }

    return results

def extract_valid_json(text: str) -> Optional[Any]:
    """
    Extract and parse the first valid JSON object found in a string.

    Handles:
    - ```json ... ``` code fences
    - extra text before/after JSON
    - malformed attempts (graceful fallback)

    Returns:
        Parsed JSON (dict/list) or None if nothing valid found.
    """

    if not text or not isinstance(text, str):
        return None

    # 🔹 1. Remove markdown code fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("```", "").strip()

    # 🔹 2. Try direct parse first (fast path)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 🔹 3. Find JSON-like blocks using bracket matching
    stack = []
    start_idx = None

    for i, char in enumerate(cleaned):
        if char == "{":
            if not stack:
                start_idx = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = cleaned[start_idx:i + 1]

                    try:
                        return json.loads(candidate)
                    except Exception:
                        # keep searching if this block is invalid
                        continue

    return None

def compute_sovereignty_score(
    hf_model: dict | None,
    web_docs: list[dict] | None = None,
    model_name: str | None = None,
    use_llm_web: bool = False,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Combine HF and optional web-based scores into a single 0-100 sovereignty score
    and per-category breakdown (0-1).

    Returns (sovereignty_score_0_100, category_scores_dict).
    """
    weights = weights or DEFAULT_WEIGHTS
    hf_scores = score_from_huggingface(hf_model or {})

    evidence_map = {}

    if use_llm_web and web_docs and model_name:
        web_scores = score_from_web_docs(web_docs, model_name, weights)

        if web_scores:
            for c in CATEGORIES:
                web_entry = web_scores.get(c, {})
                web_score = web_entry.get("score", 0.5)

                # ✅ correct blending
                hf_scores[c] = 0.5 * hf_scores[c] + 0.5 * web_score

                # ✅ keep evidence
                evidence_map[c] = web_entry.get("evidence", [])
    elif web_docs and model_name:
        # Keyword boost from web docs (no LLM)
        text = " ".join(d.get("extracted", "") for d in web_docs).lower()
        if "sovereign" in text or "sovereignty" in text or "public institution" in text:
            hf_scores["Is the model trained locally?"] = min(1.0, hf_scores["Is the model trained locally?"] + 0.15)
        if "open data" in text or ("training data" in text and "transparent" in text):
            hf_scores["Is the training data private?"] = max(0, hf_scores["Is the training data private?"] - 0.2)

    weighted = sum(hf_scores.get(c, 0.5) * weights.get(c, 1.0 / len(CATEGORIES)) for c in CATEGORIES)
    overall = round(weighted * 100, 2)
    return overall, hf_scores, evidence_map

def find_country(hf_model: dict) -> str:
    """
    Finds which country developed the model (or where the origansiation is based from)
    """
    author = (hf_model.get("author") or "").lower()
    card_data = str(hf_model.get("cardData") or "").lower()
    tags = [t.lower() for t in (hf_model.get("tags") or [])]

    # Keywords mapping (can be extended)
    COUNTRY_KEYWORDS = {
        # 🇨🇭 Switzerland
        "swiss": "Switzerland",
        "switzerland": "Switzerland",

        # 🇸🇪 Sweden
        "sweden": "Sweden",
        "ai-sweden": "Sweden",

        # 🇸🇬 Singapore
        "aisingapore": "Singapore",
        "ai singapore": "Singapore",
        "singapore": "Singapore",
        "sea-lion": "Singapore",

        # 🇬🇧 United Kingdom
        "uk": "United Kingdom",
        "united kingdom": "United Kingdom",
        "britain": "United Kingdom",
        "ucl": "United Kingdom",
        "oxford": "United Kingdom",
        "cambridge": "United Kingdom",

        # 🇮🇱 Israel
        "dicta-il": "Israel",
        "israel": "Israel",

        # 🇪🇺 European Union
        "utter-project": "European Union",
        "eu": "European Union",
        "european union": "European Union",

        # 🇫🇷 France
        "mistral": "France",
        "huggingface": "France",
        "lighton": "France",
        "france": "France",

        # 🇺🇸 United States
        "allenai": "United States",
        "openai": "United States",
        "anthropic": "United States",
        "meta": "United States",
        "google": "United States",
        "deepmind": "United States",
        "microsoft": "United States",
        "amazon": "United States",
        "aws": "United States",
        "nvidia": "United States",
        "xai": "United States",

        # 🇨🇳 China
        "deepseek": "China",
        "qwen": "China",
        "alibaba": "China",
        "baidu": "China",
        "ernie": "China",
        "pangu": "China",
        "huawei": "China",
        "zhipu": "China",
        "chatglm": "China",
        "giga-llm": "China",

        # 🇯🇵 Japan
        "yamnet": "Japan",
        "jaist": "Japan",
        "riken": "Japan",
        "fugaku": "Japan",

        # 🇰🇷 South Korea
        "naver": "South Korea",
        "hyperclova": "South Korea",
        "kakao": "South Korea",
        "skt": "South Korea",

        # 🇮🇳 India
        "sarvam": "India",
        "ai4bharat": "India",
        "india": "India",

        # 🇦🇪 United Arab Emirates
        "falcon": "United Arab Emirates",
        "tii": "United Arab Emirates",
        "technology innovation institute": "United Arab Emirates",
        "mbzuai": "United Arab Emirates",

        # 🇸🇦 Saudi Arabia
        "allam": "Saudi Arabia",
        "sdaia": "Saudi Arabia",

        # 🇷🇺 Russia
        "gigachat": "Russia",
        "yandex": "Russia",
        "yalm": "Russia",

        # 🇹🇼 Taiwan
        "taide": "Taiwan",
        "narlabs": "Taiwan",

        # 🌍 Multinational / Open
        "stability": "United Kingdom",  # Stability AI (UK-based)
        "eleutherai": "United States",
    }

    # Search heuristics in local metadata first
    joined = f"{author} {card_data} {' '.join(tags)}".lower()
    for kw, country in COUNTRY_KEYWORDS.items():
        if kw in joined:
            return country

    # Try to infer from author if it's an org name like "swiss-ai"
    for kw, country in COUNTRY_KEYWORDS.items():
        if kw in author:
            return country

    # Fallback to web search if possible
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

    if DDGS is not None and author:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(author, max_results=10))
            text_blob = " ".join((r.get("body") or "") + " " + (r.get("title") or "") for r in results).lower()
            for kw, country in COUNTRY_KEYWORDS.items():
                if kw in text_blob:
                    return country
            # Heuristic scan for known patterns
            if "switzerland" in text_blob: return "Switzerland"
            if "sweden" in text_blob: return "Sweden"
            if "singapore" in text_blob: return "Singapore"
            if "france" in text_blob: return "France"
            if "european union" in text_blob or "europe" in text_blob: return "European Union"
            if "united kingdom" in text_blob or "britain" in text_blob: return "United Kingdom"
            if "united states" in text_blob or "usa" in text_blob: return "United States"
            if "china" in text_blob: return "China"
            if "japan" in text_blob: return "Japan"
            if "uae" in text_blob or "emirates" in text_blob: return "United Arab Emirates"
            if "israel" in text_blob: return "Israel"
        except Exception:
            pass

    # Default if nothing found
    return "–"

def sort_organisation(hf_model: dict) -> str:
    """
    Sort the organisation of a model into one of the following categories:
    - Big Tech
    - State-backed
    - Independent
    - Other
    """
    author = (hf_model.get("author") or "").lower()
    tags = [t.lower() for t in (hf_model.get("tags") or [])]
    card_data = str(hf_model.get("cardData") or "").lower()
    # Use DuckDuckGo Search (ddgs) to assess the organisation type.
    # We'll use the 'duckduckgo-search' package (ddgs) if available.
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        # ddgs not installed; fallback to simple rules below.
        DDGS = None

    search_query = author
    org_type = "Independent"
    if DDGS is not None and search_query:
        # Try to infer org category from web search results.
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=10))
            text_blob = " ".join(r.get("body", "") or r.get("title", "") or "" for r in results).lower()
            # Big Tech
            if any(kw in text_blob for kw in [
                "openai", "google", "deepmind", "alphabet", "microsoft", "azure", "meta", "facebook", "amazon",
                "aws", "anthropic", "baidu", "alibaba", "tencent", "bytedance", "nvidia"
            ]):
                org_type = "Big Tech"
            # State-backed/state orgs/universities/government labs
            elif any(kw in text_blob for kw in [
                "university", "universität", "government", "federal", "ministry", "public institution",
                "swiss-ai", "ai sweden", "public ai", "national lab", "research council", "erc", "european commission"
            ]):
                org_type = "State-backed"
            # Nonprofit
            elif any(kw in text_blob for kw in [
                "non-profit", "nonprofit", "charity", "foundation", "ngo"
            ]):
                org_type = "Non-profit"
            # Community
            elif any(kw in text_blob for kw in [
                "community", "collective", "open-source"
            ]):
                org_type = "Community"
            else:
                org_type = "Independent"
        except Exception:
            # In case of search or other errors, fallback
            org_type = "Independent"
    else:
        # Fallback: Heuristic string matching
        if any(x in author for x in ORG_BIG_TECH_HINTS):
            org_type = "Big Tech"
        elif any(x in author for x in ORG_PUBLIC_HINTS):
            org_type = "State-backed"
        elif "nonprofit" in author or "non-profit" in author:
            org_type = "Non-profit"
        elif "community" in author or "collective" in author:
            org_type = "Community"
        else:
            org_type = "Independent"
    return org_type

def explain_sovereignty_score(
    evaluation: dict,
    weights: dict[str, float] | None = None,
    user_agent: str | None = None,
) -> str:
    """
    Uses PublicAI to generate a natural explanation of how the sovereignty score was derived.
    Falls back to deterministic explanation if API fails.
    """
    if not evaluation:
        return "No sovereignty evaluation available."

    weights = weights or DEFAULT_WEIGHTS
    model_id = evaluation.get("model_id", "unknown model")
    overall = evaluation.get("value")
    categories = evaluation.get("categories") or {}
    meta = evaluation.get("metadata") or {}
    evidence = evaluation.get("evidence") or {}

    if overall is None:
        return f"Sovereignty score for {model_id} could not be computed."

    # Build structured breakdown
    dims = []
    for category in CATEGORIES:
        v = float(categories.get(category, 0.5) or 0.5)
        w = float(weights.get(category, 1.0 / len(CATEGORIES)))
        dims.append(
            {
                "category": category,
                "value": round(v, 3),
                "weight": round(w, 3),
                "contribution": round(v * w, 3),
            }
        )

    uses_web = bool(meta.get("uses_web"))
    uses_llm_web = bool(meta.get("uses_llm_web"))

    # Compact evidence for the model: category → quotes + per-quote rationale
    evidence_blocks = []
    for category in CATEGORIES:
        items = evidence.get(category) or []
        if not items:
            continue
        lines = [f"Category: {category} (score in breakdown: {float(categories.get(category, 0.5) or 0.5):.3f})"]
        for ev in items[:3]:
            q = (ev.get("quote") or "").strip()
            if len(q) > 280:
                q = q[:277] + "..."
            r = (ev.get("rationale") or "").strip()
            u = (ev.get("url") or "").strip()
            if not q:
                continue
            lines.append(f'  Quote: "{q}"')
            if r:
                lines.append(f"  Why this matters for the score: {r}")
            if u:
                lines.append(f"  URL: {u}")
        if len(lines) > 1:
            evidence_blocks.append("\n".join(lines))

    evidence_section = (
        "\n\n---\n\n".join(evidence_blocks)
        if evidence_blocks
        else "(No per-category web quotes available; rely on category scores only.)"
    )

    # Prompt for LLM
    prompt = f"""
    You are explaining an AI sovereignty score to a technical user.

    Model: {model_id}
    Overall score: {overall:.2f}/100
    Used web evidence in scoring: {uses_web} (LLM over web: {uses_llm_web})

    Category scores (0 = least sovereign on that dimension, 1 = most sovereign):
    {dims}

    Web evidence (verified excerpts and why each was tied to the category score):
    {evidence_section}

    Write a concise explanation that:

    1. States WHY this model landed near this overall score (not how the formula works).
    2. Names the TOP 2–3 categories that most increased the score and the TOP 2–3 that most decreased it, using the numeric values above.
    3. Where web quotes exist, you MUST weave them in: for each category you discuss that has quotes, explain in plain language how the quoted wording supports that category's score (higher vs lower). Make the causal link explicit: "this passage suggests X, which raises/lowers sovereignty on [category] because…"
    4. If a category has no quote, infer only from the category score and do not invent citations.
    5. Avoid generic tutorials about weighted averages. Do not repeat the long boilerplate sentence about "evaluating different categories from 0 to 1".
    
    Write this in paragraph form. Your response will be displayed as a single paragraph, and ny newlines or carrage returns will includes. Similarly any italics and bolding will be ignored, so do NOT include them.
    Be specific and comparative. Prefer short quoted phrases over long pastes.
    """

    # Try LLM explanation
    try:
        return ask_publicai(
            prompt=prompt,
            user_agent=user_agent or "Sovereignty-Pipeline/1.0",
        )
    except Exception:
        weighted_avg = sum(d["contribution"] for d in dims)
        parts = [
            f"For {model_id}, the sovereignty score is {overall:.2f}/100 "
            f"(weighted blend of category scores, weighted average of contributions ≈ {weighted_avg:.2f} on a 0–1 scale before scaling to 0–100)."
        ]
        if evidence_blocks:
            parts.append("Evidence-linked notes:")
            for category in CATEGORIES:
                items = evidence.get(category) or []
                for ev in items[:2]:
                    q = (ev.get("quote") or "").strip()
                    r = (ev.get("rationale") or "").strip()
                    if not q:
                        continue
                    short_q = q if len(q) <= 200 else q[:197] + "..."
                    line = f"- [{category}] \"{short_q}\""
                    if r:
                        line += f" → {r}"
                    parts.append(line)
        else:
            parts.append(
                "Higher-scoring categories increased the result; lower-scoring categories pulled it down."
            )
        return " ".join(parts)

def score_organisation_from_metadata(hf_model: dict) -> tuple[float, dict[str, float]]:
    """
    Very simple, metadata-only organisation sovereignty heuristic.

    It looks at the model's `author`, tags and cardData text to guess:
    - whether funding is likely public vs purely private
    - whether the org is independent from big tech platforms
    - whether there is an explicit local / sovereign focus
    """
    author = (hf_model.get("author") or "").lower()
    tags = [t.lower() for t in (hf_model.get("tags") or [])]
    card_data = str(hf_model.get("cardData") or "").lower()
    text = " ".join([author] + tags + [card_data])

    scores = {
        "public_funding_score": 0.5,
        "independence_from_bigtech_score": 0.5,
        "local_sovereignty_focus_score": 0.5,
    }

    has_public_hint = any(h in text for h in ORG_PUBLIC_HINTS)
    has_bigtech_hint = any(h in text for h in ORG_BIG_TECH_HINTS)

    if has_public_hint:
        scores["public_funding_score"] = 0.8
        scores["independence_from_bigtech_score"] = max(
            scores["independence_from_bigtech_score"], 0.7
        )

    if has_bigtech_hint:
        scores["independence_from_bigtech_score"] = 0.3
        if not has_public_hint:
            scores["public_funding_score"] = 0.3

    # Local / sovereign focus: look for country / sovereignty language
    if any(
        kw in text
        for kw in [
            "swiss",
            "switzerland",
            "sweden",
            "uk ",
            "britain",
            "europe",
            "eu ",
            "local",
            "sovereign",
            "sovereignty",
            "public infrastructure",
        ]
    ):
        scores["local_sovereignty_focus_score"] = 0.8

    overall = round(
        (scores["public_funding_score"]
         + scores["independence_from_bigtech_score"]
         + scores["local_sovereignty_focus_score"])
        / 3.0
        * 100,
        2,
    )
    return overall, scores


def evaluate_model_for_hf(
    model_id: str,
    use_web: bool = False,
    use_llm_web: bool = False,
) -> dict:
    """
    Convenience helper for publishing a Hugging Face evaluation result.

    Given a Hugging Face model ID (e.g. 'swiss-ai/Apertus-70B-2509'), this:
    - fetches model metadata from the Hub
    - optionally collects web evidence (if use_web=True and optional deps installed)
    - optionally uses an LLM over the web docs (if use_llm_web=True and PUBLICAI_KEY is set)
    - returns a JSON-serialisable dict you can paste into a model card.

    This does NOT call any Hugging Face-specific APIs; it just returns data.
    """
    from pipeline.sources import fetch_huggingface_model, fetch_web_evidence

    hf_model = fetch_huggingface_model(model_id)
    # Use last segment as human-readable name for prompts
    model_name = model_id.split("/")[-1]
    # HF metadata can be missing if the model id is invalid; be defensive.
    author = ((hf_model or {}).get("author") or "").lower()

    # Defaults if we can't infer org/country from HF metadata.
    org_type = "Independent"
    country = "–"

    web_docs = []
    if use_web:
        # Best-effort; if optional deps are missing, this will just return [].
        web_docs = fetch_web_evidence(model_name, top_k_per_query=3, delay_between_requests=1.0, verbose=False)
        if hf_model:
            org_type = sort_organisation(hf_model)
            country = find_country(hf_model)

    sources = sorted({d.get("url") for d in (web_docs or []) if d.get("url")})

    overall, categories, evidence = compute_sovereignty_score(
        hf_model,
        web_docs=web_docs or None,
        model_name=model_name,
        use_llm_web=use_llm_web,
    )
    if use_web:
        return {
            "model_id": model_id,
            "author": author,
            "metric_name": "sovereignty_score",
            "metric_type": "custom",
            "value": overall,
            "categories": categories,
            "sources": sources,
            "organisation_type": org_type,
            "country": country,
            "evidence": evidence,
            "metadata": {
                "source": "public-ai sovereignty pipeline",
                "version": "0.1.0",
                "uses_web": bool(web_docs),
                "requested_use_web": use_web,
                "uses_llm_web": bool(use_llm_web),
            },
        }
    else:
        return {
            "model_id": model_id,
            "author": author,
            "metric_name": "sovereignty_score",
            "metric_type": "custom",
            "organisation_type": org_type,
            "country": country,
            "value": overall,
            "categories": categories,
            "sources": sources,
            "metadata": {
                "source": "public-ai sovereignty pipeline",
                "version": "0.1.0",
                "uses_web": bool(web_docs),
                "requested_use_web": use_web,
                "uses_llm_web": bool(use_llm_web),
            }, 
        }

def compare_with_neighbors(target_model_id: str, all_models: list[dict], k: int = 3):
    """
    Returns nearest models by score for local comparison.
    """
    target = next((m for m in all_models if m["model_id"] == target_model_id), None)
    if not target:
        return []

    target_score = target.get("value", 0)

    sorted_models = sorted(
        all_models,
        key=lambda m: abs(m.get("value", 0) - target_score)
    )

    neighbors = [m for m in sorted_models if m["model_id"] != target_model_id][:k]

    return [
        {
            "model_id": m["model_id"],
            "score": m["value"],
            "categories": m.get("categories", {})
        }
        for m in neighbors
    ]

def evaluate_organisation_for_hf(
    model_id: str,
) -> dict:
    """
    Evaluate the *organisation* behind a model, separately from the model itself.

    This currently uses **only Hugging Face metadata** (author, tags, cardData)
    and simple string heuristics. It does not scrape the web and does not try
    to trace funding sources precisely; instead it estimates:

    - public_funding_score (0–1)
    - independence_from_bigtech_score (0–1)
    - local_sovereignty_focus_score (0–1)
    - overall organisation sovereignty score (0–100)
    """
    from pipeline.sources import fetch_huggingface_model

    hf_model = fetch_huggingface_model(model_id)
    if not hf_model:
        return {
            "model_id": model_id,
            "organisation": None,
            "metric_name": "organisation_sovereignty_score",
            "metric_type": "custom",
            "value": None,
            "dimensions": {},
            "metadata": {
                "source": "public-ai organisation sovereignty heuristic",
                "version": "0.1.0",
                "note": "no Hugging Face metadata available for this model_id",
            },
        }

    author = hf_model.get("author")
    overall, dims = score_organisation_from_metadata(hf_model)

    return {
        "model_id": model_id,
        "organisation": author,
        "metric_name": "organisation_sovereignty_score",
        "metric_type": "custom",
        "value": overall,
        "dimensions": dims,
        "metadata": {
            "source": "public-ai organisation sovereignty heuristic",
            "version": "0.1.0",
        },
    }

def build_summary_stats(all_models: list[dict]):
    scores = sorted(m["value"] for m in all_models if m.get("value") is not None)

    if not scores:
        return {}

    median = scores[len(scores)//2]

    return {
        "min": min(scores),
        "max": max(scores),
        "median": median,
        "models": [
            {
                "model_id": m["model_id"],
                "score": m["value"]
            }
            for m in all_models
        ]
    }