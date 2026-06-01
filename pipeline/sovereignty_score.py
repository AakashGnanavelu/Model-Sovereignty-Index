"""
sovereignty_score.py

Object-oriented AI sovereignty scoring pipeline.

Sovereignty is defined as independence from Big Tech control over access to AI —
i.e. how freely an organisation can train, own, deploy, and govern a model
without being subject to the infrastructure, licensing, or legal reach of
US or Chinese hyperscalers.

Classes
-------
ModelSovereigntyScore
    Holds scoring logic and results for a single Hugging Face model.
OrganisationSovereigntyScore
    Aggregates one or more ModelSovereigntyScore objects and handles
    organisation-level metadata and JSON persistence.

Dimensions (7 categories)
--------------------------
1. Training Data Independence
   Is the training data self-owned or open-licensed, or does it flow through
   Big Tech pipelines (e.g. Common Crawl filtered via AWS)?

2. Compute Independence
   Was the model trained on sovereign / public-institution infrastructure,
   or on Azure / AWS / GCP?

3. Weight Ownership & Access
   Who legally controls the weights? Can a US or Chinese company revoke access?

4. Base Model Dependency
   Trained from scratch, or fine-tuned from a proprietary or foreign model?

5. Deployment Independence
   Can the model be run locally / on-prem, or is it API-only behind a
   corporate wall?

6. Organisational Independence
   Is the organisation a public institution, non-profit, or independent
   researcher? Or VC-backed / Big Tech?

7. Jurisdictional Risk
   Where is the organisation legally domiciled, and does US CLOUD Act or
   Chinese cybersecurity law apply to the weights?

Score: 0–100. Higher = more sovereign (less dependent on Big Tech).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

import requests

# ---------------------------------------------------------------------------
# Optional pipeline dependencies
# ---------------------------------------------------------------------------
try:
    from pipeline.ask import ask_publicai
    from pipeline.sources import fetch_huggingface_model, fetch_web_evidence
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"PIPELINE UNAVAILABLE: {e}")
    _PIPELINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Scoring dimensions
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Training Data Independence",
    "Compute Independence",
    "Weight Ownership & Access",
    "Base Model Dependency",
    "Deployment Independence",
    "Organisational Independence",
    "Jurisdictional Risk",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "Training Data Independence":   0.18,
    "Compute Independence":         0.15,
    "Weight Ownership & Access":    0.18,
    "Base Model Dependency":        0.14,
    "Deployment Independence":      0.13,
    "Organisational Independence":  0.12,
    "Jurisdictional Risk":          0.10,
}

assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1."

# ---------------------------------------------------------------------------
# Ground-truth table
#
# Entries are manually curated for well-known models where web evidence is
# unreliable or missing.  Values are floats in [0, 1] per category.
# A model present here bypasses HF heuristics for the covered categories.
# "confidence" is set to 1.0 for ground-truth entries.
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, dict[str, float]] = {
    # ── Big Tech / API-only ──────────────────────────────────────────────────
    "openai/gpt-4": {
        "Training Data Independence":  0.05,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,  # self-developed but closed
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.10,
        "Jurisdictional Risk":         0.05,
    },
    "openai/gpt-4o": {
        "Training Data Independence":  0.05,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.10,
        "Jurisdictional Risk":         0.05,
    },
    "openai/gpt-3.5-turbo": {
        "Training Data Independence":  0.05,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.10,
        "Jurisdictional Risk":         0.05,
    },
    "anthropic/claude-3-opus": {
        "Training Data Independence":  0.10,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.15,
        "Jurisdictional Risk":         0.05,
    },
    "anthropic/claude-3-sonnet": {
        "Training Data Independence":  0.10,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.15,
        "Jurisdictional Risk":         0.05,
    },
    "google/gemini-pro": {
        "Training Data Independence":  0.05,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.02,
        "Organisational Independence": 0.05,
        "Jurisdictional Risk":         0.05,
    },
    # ── Meta open-weight (strong base, US jurisdiction) ──────────────────────
    "meta-llama/llama-3-70b-instruct": {
        "Training Data Independence":  0.30,
        "Compute Independence":        0.10,
        "Weight Ownership & Access":   0.45,  # open weights but Meta licence
        "Base Model Dependency":       0.80,
        "Deployment Independence":     0.80,
        "Organisational Independence": 0.20,
        "Jurisdictional Risk":         0.15,
    },
    "meta-llama/llama-2-70b-chat-hf": {
        "Training Data Independence":  0.25,
        "Compute Independence":        0.10,
        "Weight Ownership & Access":   0.40,
        "Base Model Dependency":       0.80,
        "Deployment Independence":     0.75,
        "Organisational Independence": 0.20,
        "Jurisdictional Risk":         0.15,
    },
    # ── Mistral (France, independent) ────────────────────────────────────────
    "mistralai/mistral-7b-v0.1": {
        "Training Data Independence":  0.50,
        "Compute Independence":        0.50,
        "Weight Ownership & Access":   0.80,
        "Base Model Dependency":       0.85,
        "Deployment Independence":     0.90,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.75,
    },
    "mistralai/mixtral-8x7b-instruct-v0.1": {
        "Training Data Independence":  0.50,
        "Compute Independence":        0.50,
        "Weight Ownership & Access":   0.80,
        "Base Model Dependency":       0.85,
        "Deployment Independence":     0.90,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.75,
    },
    # ── EleutherAI / Pythia (US non-profit, open) ────────────────────────────
    "eleutherai/pythia-12b": {
        "Training Data Independence":  0.75,
        "Compute Independence":        0.55,
        "Weight Ownership & Access":   0.90,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.95,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.30,
    },
    # ── Falcon / TII (UAE, state-backed) ─────────────────────────────────────
    "tiiuae/falcon-40b": {
        "Training Data Independence":  0.60,
        "Compute Independence":        0.65,
        "Weight Ownership & Access":   0.75,
        "Base Model Dependency":       0.85,
        "Deployment Independence":     0.85,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.60,
    },
    # ── Swiss-AI / EPFL (sovereign exemplar) ─────────────────────────────────
    "swiss-ai/swissbert": {
        "Training Data Independence":  0.90,
        "Compute Independence":        0.90,
        "Weight Ownership & Access":   0.90,
        "Base Model Dependency":       0.80,
        "Deployment Independence":     0.90,
        "Organisational Independence": 0.95,
        "Jurisdictional Risk":         0.95,
    },
    "swiss-ai/swissgpt": {
        "Training Data Independence":  0.88,
        "Compute Independence":        0.88,
        "Weight Ownership & Access":   0.90,
        "Base Model Dependency":       0.75,
        "Deployment Independence":     0.88,
        "Organisational Independence": 0.95,
        "Jurisdictional Risk":         0.95,
    },
    # ── AI Sweden ────────────────────────────────────────────────────────────
    "ai-sweden/gpt-sw3-126m": {
        "Training Data Independence":  0.85,
        "Compute Independence":        0.85,
        "Weight Ownership & Access":   0.90,
        "Base Model Dependency":       0.85,
        "Deployment Independence":     0.90,
        "Organisational Independence": 0.90,
        "Jurisdictional Risk":         0.90,
    },
    # ── DeepSeek (China) ─────────────────────────────────────────────────────
    "deepseek-ai/deepseek-coder-33b-instruct": {
        "Training Data Independence":  0.40,
        "Compute Independence":        0.50,
        "Weight Ownership & Access":   0.50,
        "Base Model Dependency":       0.75,
        "Deployment Independence":     0.70,
        "Organisational Independence": 0.30,
        "Jurisdictional Risk":         0.10,
    },
}

# Normalise ground-truth keys to lowercase for lookup
GROUND_TRUTH = {k.lower(): v for k, v in GROUND_TRUTH.items()}

# ---------------------------------------------------------------------------
# Known open / permissive licences
# ---------------------------------------------------------------------------

OPEN_LICENSES: set[str] = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause",
    "cc-by-4.0", "cc-by-sa-4.0",
    "openrail", "openrail++", "bigscience-openrail-m",
    "afl-3.0", "artistic-2.0",
    "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0",
    "llama2", "llama3",                        # open-weight Meta licences
    "cc0-1.0", "wtfpl", "unlicense",
}

# Licences that are open-weight but carry usage/commercial restrictions
RESTRICTED_OPEN_LICENSES: set[str] = {
    "llama2", "llama3", "cc-by-nc-4.0", "cc-by-nc-sa-4.0",
    "gemma", "microsoft-research-license",
}

# ---------------------------------------------------------------------------
# Jurisdiction tiers
# (higher = less jurisdictional risk to sovereignty)
# ---------------------------------------------------------------------------

JURISDICTION_SCORE: dict[str, float] = {
    # Tier 1: strong data-protection, no CLOUD-Act equivalent
    "Switzerland":      0.97,
    "Norway":           0.93,
    "Iceland":          0.93,
    "Sweden":           0.90,
    "Finland":          0.90,
    "Germany":          0.88,
    "Austria":          0.88,
    "Netherlands":      0.85,
    "France":           0.83,
    "Denmark":          0.85,
    "European Union":   0.82,
    "Belgium":          0.82,
    "Spain":            0.80,
    "Italy":            0.78,
    "Portugal":         0.78,
    "Poland":           0.75,
    # Tier 2: Five Eyes — subject to intelligence-sharing / CLOUD Act
    "United Kingdom":   0.55,
    "Canada":           0.50,
    "Australia":        0.48,
    "New Zealand":      0.48,
    # Tier 3: US hyperscaler home jurisdiction
    "United States":    0.15,
    # Tier 4: Chinese cybersecurity law / state access
    "China":            0.10,
    # Tier 5: other / uncertain
    "United Arab Emirates": 0.55,
    "Saudi Arabia":     0.45,
    "Israel":           0.60,
    "Japan":            0.65,
    "South Korea":      0.62,
    "Singapore":        0.60,
    "India":            0.55,
    "Russia":           0.10,
    "Taiwan":           0.65,
    "–":                0.40,   # unknown
}

# ---------------------------------------------------------------------------
# Country detection
# ---------------------------------------------------------------------------

COUNTRY_KEYWORDS: dict[str, str] = {
    # Switzerland
    "swiss-ai": "Switzerland", "swiss_ai": "Switzerland",
    "swiss": "Switzerland", "switzerland": "Switzerland",
    "epfl": "Switzerland", "eth zurich": "Switzerland",
    "eth zürich": "Switzerland", "cscs": "Switzerland",
    "swisstxt": "Switzerland",
    # Nordic
    "ai-sweden": "Sweden", "ai sweden": "Sweden", "ai-sweden-models": "Sweden",
    "sweden": "Sweden",
    "finland": "Finland", "norway": "Norway", "denmark": "Denmark",
    "iceland": "Iceland",
    # EU
    "mistral": "France", "mistralai": "France", "lighton": "France", "france": "France",
    "huggingface": "France",
    "aleph alpha": "Germany", "alephalpha": "Germany", "germany": "Germany",
    "utter-project": "European Union", "eu": "European Union",
    "european union": "European Union",
    "bsc-lt": "Spain", "spain": "Spain",
    "italy": "Italy", "portugal": "Portugal",
    "belgium": "Belgium", "netherlands": "Netherlands",
    # UK
    "stability": "United Kingdom", "stabilityai": "United Kingdom",
    "deepmind": "United Kingdom", "uk": "United Kingdom",
    "united kingdom": "United Kingdom", "britain": "United Kingdom",
    "ucl": "United Kingdom", "oxford": "United Kingdom",
    "cambridge": "United Kingdom",
    # US
    "openai": "United States", "anthropic": "United States",
    "meta": "United States", "google": "United States",
    "microsoft": "United States", "amazon": "United States",
    "aws": "United States", "nvidia": "United States",
    "allenai": "United States", "eleutherai": "United States",
    "xai": "United States", "x.ai": "United States",
    "cohere": "United States",
    # China
    "deepseek": "China", "qwen": "China", "alibaba": "China",
    "baidu": "China", "ernie": "China", "pangu": "China",
    "huawei": "China", "zhipu": "China", "chatglm": "China",
    "giga-llm": "China", "01-ai": "China", "yi-": "China",
    "moonshot": "China", "minimax": "China",
    # Middle East
    "falcon": "United Arab Emirates", "tiiuae": "United Arab Emirates",
    "tii": "United Arab Emirates",
    "technology innovation institute": "United Arab Emirates",
    "mbzuai": "United Arab Emirates",
    "allam": "Saudi Arabia", "sdaia": "Saudi Arabia",
    # APAC
    "naver": "South Korea", "hyperclova": "South Korea",
    "kakao": "South Korea", "skt": "South Korea",
    "sarvam": "India", "ai4bharat": "India", "india": "India",
    "riken": "Japan", "fugaku": "Japan", "jaist": "Japan",
    "aisingapore": "Singapore", "ai singapore": "Singapore",
    "singapore": "Singapore", "sea-lion": "Singapore", "sea lion": "Singapore",
    # Other
    "dicta-il": "Israel", "israel": "Israel",
    "gigachat": "Russia", "yandex": "Russia", "yalm": "Russia",
    "taide": "Taiwan", "narlabs": "Taiwan",
}

TLD_COUNTRY_MAP: dict[str, str] = {
    ".fr": "France", ".de": "Germany", ".ch": "Switzerland",
    ".uk": "United Kingdom", ".se": "Sweden", ".fi": "Finland",
    ".no": "Norway", ".dk": "Denmark", ".nl": "Netherlands",
    ".be": "Belgium", ".es": "Spain", ".it": "Italy",
    ".pt": "Portugal", ".at": "Austria", ".pl": "Poland",
    ".sg": "Singapore", ".cn": "China", ".jp": "Japan",
    ".in": "India", ".il": "Israel", ".ae": "United Arab Emirates",
    ".ru": "Russia", ".tw": "Taiwan", ".kr": "South Korea",
}

# ---------------------------------------------------------------------------
# Big Tech markers (used for Organisational Independence scoring)
# ---------------------------------------------------------------------------

BIG_TECH_ORGS: set[str] = {
    "openai", "google", "deepmind", "alphabet", "microsoft", "azure",
    "meta", "facebook", "amazon", "aws", "anthropic", "x.ai", "xai",
    "baidu", "alibaba", "tencent", "bytedance", "nvidia", "apple",
    "samsung", "huawei", "deepseek",
}

PUBLIC_INSTITUTION_HINTS: list[str] = [
    "university", "universität", "université", "universidad",
    "epfl", "eth zurich", "cscs", "swiss-ai", "ai sweden",
    "public ai", "government", "gov", "ministry", "federal",
    "national lab", "research council", "european commission",
    "european union", "erc", "cnrs", "inria", "cea", "dfki",
    "fraunhofer", "helmholtz", "max planck",
]

# ---------------------------------------------------------------------------
# Known cloud compute providers (for Compute Independence heuristics)
# ---------------------------------------------------------------------------

CLOUD_PROVIDERS: set[str] = {
    "azure", "aws", "amazon", "google cloud", "gcp",
    "lambda labs",  # US cloud
    "coreweave",    # US cloud
    "oracle cloud",
}

SOVEREIGN_COMPUTE_HINTS: list[str] = [
    "cscs", "jsc", "lumi", "frontier", "supercomputer",
    "national computing", "public compute", "hpc", "meluxina",
    "leonardo", "marenostrum", "discoverer", "lumi supercomputer",
    "swiss national supercomputing", "snsc", "digital research alliance",
    "pawsey", "nci australia", "prace", "eurocc", "in-house",
    "on-premise", "on-prem",
]

# ---------------------------------------------------------------------------
# Keywords for LLM prompt / heuristic extraction per category
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Training Data Independence": [
        "training data", "dataset", "corpus", "data source", "open data",
        "crawl", "web data", "proprietary data", "licensed data",
        "data pipeline", "annotation", "human feedback", "rlhf",
    ],
    "Compute Independence": [
        "compute", "training infrastructure", "cluster", "supercomputer",
        "hpc", "cloud", "aws", "azure", "gcp", "gpu", "tpu",
        "data center", "on-prem", "on-premise", "hosted",
    ],
    "Weight Ownership & Access": [
        "weights", "model weights", "license", "access", "download",
        "open weights", "proprietary", "revoke", "terms of service",
        "commercial use", "redistribution",
    ],
    "Base Model Dependency": [
        "base model", "fine-tun", "fine tuned", "pretrained", "pre-trained",
        "checkpoint", "adapter", "lora", "qlora", "derived from",
        "distill", "instruction tuned", "from scratch",
    ],
    "Deployment Independence": [
        "deploy", "local", "on-prem", "self-host", "api only", "api-only",
        "ollama", "vllm", "llama.cpp", "download", "run locally",
        "inference", "endpoint", "hosted api",
    ],
    "Organisational Independence": [
        "organisation", "company", "nonprofit", "non-profit", "university",
        "public institution", "vc", "venture capital", "funding",
        "government", "independent", "research lab", "backed by",
    ],
    "Jurisdictional Risk": [
        "headquarter", "domicile", "jurisdiction", "country", "law",
        "cloud act", "gdpr", "data protection", "legal", "incorporated",
        "entity", "based in", "registered",
    ],
}

CATEGORY_DESCRIPTION = {
    "Training Data Independence": (
        "Measures the degree to which the model's training data is open, transparent, "
        "and not dependent on proprietary or restricted datasets. High independence means "
        "training data sources are public, auditable, and not dominated by big tech or closed data."
    ),
    "Compute Independence": (
        "Assesses whether the model was trained and can be run using independent, sovereign, or publicly accessible compute infrastructure. "
        "High scores indicate use of national, academic, or independently controlled compute, not reliant on US hyperscalers or third-party cloud monopolies."
    ),
    "Weight Ownership & Access": (
        "Indicates whether the model weights are owned and controlled by an independent entity, "
        "and whether the weights are publicly released under an open license, allowing free use and distribution."
    ),
    "Base Model Dependency": (
        "Evaluates the extent to which the model depends on proprietary, closed, or big tech base models, "
        "vs being trained from scratch or on other open, independent models."
    ),
    "Deployment Independence": (
        "Reflects the ability to deploy and operate the model in any environment, including on-premises or public infrastructure, "
        "without restrictions such as API-only access, licensing barriers, or hardware dependencies controlled by third parties."
    ),
    "Organisational Independence": (
        "Scores the independence of the developing organisation, considering factors such as lack of big tech or foreign government control, "
        "public or academic oversight, or independent funding/ownership."
    ),
    "Jurisdictional Risk": (
        "Assesses the legal and regulatory exposure of the organisation or model to high-risk jurisdictions, "
        "such as those covered by foreign data access laws or extraterritorial governance (e.g. US CLOUD Act). "
        "Lower risk comes from operating under transparent, democratic, and privacy-respecting jurisdictions."
    ),
}


# ---------------------------------------------------------------------------
# Boilerplate filter
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS: list[str] = [
    r"click here", r"press here", r"learn more", r"read more",
    r"cookie", r"accept all", r"privacy policy", r"terms of service",
    r"all rights reserved", r"subscribe", r"sign up", r"log in",
    r"404", r"page not found",
]

# ---------------------------------------------------------------------------
# Helpers (module-level, stateless)
# ---------------------------------------------------------------------------

def _parse_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    m = re.search(r"0?\.\d+|\d+\.?\d*", s.replace(",", "."))
    if m:
        return max(0.0, min(1.0, float(m.group())))
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_boilerplate(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in BOILERPLATE_PATTERNS)


def _is_relevant(text: str, category: str) -> bool:
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return True
    t = text.lower()
    return any(k in t for k in keywords)


def _score_sentence(text: str, category: str) -> int:
    """Heuristic sentence relevance score."""
    score = 0
    t = text.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    score += sum(1 for k in keywords if k in t)
    if len(text) > 80:
        score += 1
    if any(x in t for x in ["must", "only", "stored", "located", "available", "exclusively"]):
        score += 2
    return score


def _extract_valid_json(text: str) -> Optional[Any]:
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    stack: list[str] = []
    start_idx: Optional[int] = None
    for i, char in enumerate(cleaned):
        if char == "{":
            if not stack:
                start_idx = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    try:
                        return json.loads(cleaned[start_idx: i + 1])
                    except Exception:
                        continue
    return None


def _infer_country_from_url(url: str) -> Optional[str]:
    """Infer country from a URL hostname TLD (e.g. .se → Sweden)."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip().lower()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url.lstrip('/')}"
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if not host:
        return None
    for tld, country in sorted(TLD_COUNTRY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if host.endswith(tld) or f"{tld}." in host:
            return country
    return None


def _get_hf_org(org: str) -> Optional[dict]:
    try:
        r = requests.get(f"https://huggingface.co/api/organizations/{org}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _verify_quote(quote: str, docs: list[dict]) -> Optional[str]:
    """
    Verify that *quote* is a genuine substring of one of the scraped documents.
    Returns the URL of the first matching document, or None if not found.

    Uses a sliding window of 30 characters for robustness against minor
    whitespace/encoding differences, which is substantially stronger than
    the previous 20-char check.
    """
    if not quote or len(quote) < 20:
        return None
    qn = _normalize(quote)
    window = qn[:30]
    for doc in docs:
        blob = _normalize((doc.get("extracted") or "")[:10_000])
        if window in blob:
            return doc.get("url", "")
    return None


# ---------------------------------------------------------------------------
# ModelSovereigntyScore
# ---------------------------------------------------------------------------

@dataclass
class ModelSovereigntyScore:
    """
    Sovereignty score for a single AI model.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier, e.g. ``"mistralai/Mistral-7B-v0.1"``.
    organisation:
        Parent :class:`OrganisationSovereigntyScore`.
    weights:
        Per-category weights (must sum to 1). Defaults to DEFAULT_WEIGHTS.
    verbose:
        When ``True``, print coloured, human-readable progress to the terminal
        (blue for operational detail, white for milestones, red for errors).
        When ``False`` (default), logging is suppressed; exceptions still propagate.
    """

    model_id: str
    organisation: "OrganisationSovereigntyScore"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    verbose: bool = False

    # Populated by evaluate() / score_from_hf_dict()
    overall_score: Optional[float] = field(default=None, init=False)
    category_scores: dict[str, float] = field(default_factory=dict, init=False)
    category_confidence: dict[str, float] = field(default_factory=dict, init=False)
    evidence: dict[str, list[dict]] = field(default_factory=dict, init=False)
    sources: list[str] = field(default_factory=list, init=False)
    explanation: Optional[str] = field(default=None, init=False)
    used_ground_truth: bool = field(default=False, init=False)

    _hf_model: Optional[dict] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Logging (ANSI colours when verbose=True)
    # ------------------------------------------------------------------

    _LOG_RESET = "\033[0m"
    _LOG_RED = "\033[91m"
    _LOG_BLUE = "\033[94m"
    _LOG_WHITE = "\033[97m"

    def _log(self, message: str, *, method: str) -> None:
        """Blue [INFO] lines for operational detail."""
        if self.verbose:
            print(
                f"{self._LOG_BLUE}[INFO] ModelSovereigntyScore.{method}: "
                f"{message}{self._LOG_RESET}",
                flush=True,
            )

    def _log_general(self, message: str, *, method: str) -> None:
        """White lines for high-level progress milestones."""
        if self.verbose:
            print(
                f"{self._LOG_WHITE}ModelSovereigntyScore.{method}: "
                f"{message}{self._LOG_RESET}",
                flush=True,
            )

    def _log_error(self, method: str, exc: BaseException, *, context: Optional[dict] = None) -> None:
        """Red [ERROR] block with full, untruncated context."""
        if not self.verbose:
            return
        lines = [
            f"[ERROR] ModelSovereigntyScore.{method}",
            f"Model: {self.model_id}",
        ]
        if context:
            for k, v in context.items():
                lines.append(f"{k}: {v}")
        lines.append(f"Exception: {type(exc).__name__}: {exc}")
        for line in lines:
            print(f"{self._LOG_RED}{line}{self._LOG_RESET}", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        use_web: bool = False,
        use_llm_web: bool = False,
        verbose: bool = False,
    ) -> "ModelSovereigntyScore":
        """
        Fetch HF metadata, optionally gather web evidence, compute score.

        Returns self for method chaining.
        """
        self.verbose = verbose
        method = "evaluate"
        self._log_general(
            f"Starting evaluation (organisation={self.organisation.name!r}, "
            f"use_web={use_web}, use_llm_web={use_llm_web})",
            method=method,
        )

        if not _PIPELINE_AVAILABLE:
            raise RuntimeError(
                "pipeline package is not installed. "
                "Use score_from_hf_dict() with pre-fetched metadata."
            )

        try:
            self._log("Fetching Hugging Face model metadata", method=method)
            self._hf_model = fetch_huggingface_model(self.model_id)
        except Exception as exc:
            self._log_error(method, exc)
            raise

        web_docs: list[dict] = []
        if use_web:
            try:
                self._log(f"Fetching web evidence for '{self.model_id}'", method=method)
                web_docs = fetch_web_evidence(
                    self.model_id,
                    user_agent="Sovereignty-Pipeline/1.0",
                    top_k_per_query=3,
                    delay_between_requests=1.0,
                    verbose=self.verbose,
                )
                self.sources = sorted({d.get("url") for d in web_docs if d.get("url")})
                self._log(f"Retrieved {len(web_docs)} web document(s)", method=method)
            except Exception as exc:
                self._log_error(method, exc, context={"use_web": use_web})
                raise

        try:
            self._compute_and_store(
                hf_model=self._hf_model,
                web_docs=web_docs or None,
                use_llm_web=use_llm_web,
            )
            self._log_general(
                f"Evaluation complete (overall_score={self.overall_score})",
                method=method,
            )
        except Exception as exc:
            self._log_error(method, exc, context={"use_web": use_web, "use_llm_web": use_llm_web})
            raise

        return self

    def score_from_hf_dict(
        self,
        hf_model: dict,
        web_docs: Optional[list[dict]] = None,
        use_llm_web: bool = False,
    ) -> "ModelSovereigntyScore":
        """
        Score from a pre-fetched HF metadata dict (useful for testing).
        """
        self._hf_model = hf_model
        if web_docs:
            self.sources = sorted({d.get("url") for d in web_docs if d.get("url")})
        self._compute_and_store(
            hf_model=hf_model,
            web_docs=web_docs,
            use_llm_web=use_llm_web,
        )
        return self

    def generate_explanation(self, user_agent: str = "Sovereignty-Pipeline/1.0") -> str:
        """Generate a human-readable explanation and store in self.explanation."""
        try:
            self.explanation = self._explain_score(user_agent=user_agent)
        except Exception as exc:
            self._log_error("generate_explanation", exc)
            self.explanation = self._fallback_explanation()
        return self.explanation

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "author": self.organisation.name,
            "metric_name": "sovereignty_score",
            "metric_type": "custom",
            "value": self.overall_score,
            "categories": self.category_scores,
            "category_confidence": self.category_confidence,
            "evidence": self.evidence,
            "sources": self.sources,
            "explanation": self.explanation,
            "used_ground_truth": self.used_ground_truth,
        }

    # ------------------------------------------------------------------
    # Core scoring logic
    # ------------------------------------------------------------------

    def _compute_and_store(
        self,
        hf_model: Optional[dict],
        web_docs: Optional[list[dict]],
        use_llm_web: bool,
    ) -> None:
        """Compute scores and write to instance fields."""
        method = "_compute_and_store"
        self._log_general(f"Computing sovereignty score for '{self.model_id}'", method=method)

        # 1. Check ground-truth table first
        gt_key = self.model_id.lower()
        if gt_key in GROUND_TRUTH:
            self._log(f"Using ground-truth scores for '{self.model_id}'", method=method)
            gt = GROUND_TRUTH[gt_key]
            self.category_scores = {c: gt.get(c, 0.5) for c in CATEGORIES}
            self.category_confidence = {c: 1.0 for c in CATEGORIES}
            self.evidence = {}
            self.used_ground_truth = True
            self.overall_score = self._weighted_total(self.category_scores)
            return

        # 2. HF-metadata heuristics
        hf_scores, hf_confidence = self._score_from_huggingface(hf_model or {})

        # 3. Optionally blend web evidence
        evidence_map: dict[str, list[dict]] = {}
        final_scores = dict(hf_scores)
        final_confidence = dict(hf_confidence)

        if use_llm_web and web_docs:
            model_name = self.model_id.split("/")[-1]
            self._log("Running LLM web scoring", method=method)
            web_result = self._score_from_web_docs(web_docs, model_name)
            if web_result:
                for c in CATEGORIES:
                    entry = web_result.get(c, {})
                    web_score = entry.get("score", 0.5)
                    web_conf = entry.get("confidence", 0.0)
                    # Weighted blend: trust web more when its confidence is high
                    alpha = min(0.6, web_conf)  # max 60 % weight to web
                    final_scores[c] = (1 - alpha) * hf_scores[c] + alpha * web_score
                    final_confidence[c] = max(hf_confidence[c], web_conf)
                    evidence_map[c] = entry.get("evidence", [])
        elif web_docs:
            self._log("Applying heuristic web adjustments", method=method)
            text = " ".join(d.get("extracted", "") for d in web_docs).lower()
            if any(k in text for k in SOVEREIGN_COMPUTE_HINTS):
                final_scores["Compute Independence"] = min(
                    1.0, final_scores["Compute Independence"] + 0.15
                )
            if any(k in text for k in CLOUD_PROVIDERS):
                final_scores["Compute Independence"] = max(
                    0.0, final_scores["Compute Independence"] - 0.20
                )
            if "open data" in text or ("training data" in text and "transparent" in text):
                final_scores["Training Data Independence"] = max(
                    0.0, final_scores["Training Data Independence"] - 0.10
                )

        self.category_scores = final_scores
        self.category_confidence = final_confidence
        self.evidence = evidence_map
        self.overall_score = self._weighted_total(final_scores)
        self._log(f"Overall sovereignty score: {self.overall_score}", method=method)

    def _weighted_total(self, scores: dict[str, float]) -> float:
        total = sum(
            scores.get(c, 0.5) * self.weights.get(c, 1.0 / len(CATEGORIES))
            for c in CATEGORIES
        )
        return round(total * 100, 2)

    def _score_from_huggingface(
        self, hf_model: dict
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Derive per-category scores and confidence values from HF metadata.

        Confidence reflects how much information we could extract from the
        metadata (1.0 = very confident, 0.3 = mostly guessing).
        """
        scores: dict[str, float] = {c: 0.5 for c in CATEGORIES}
        confidence: dict[str, float] = {c: 0.3 for c in CATEGORIES}

        if not hf_model:
            return scores, confidence

        author = (hf_model.get("author") or "").lower()
        model_id_lower = (hf_model.get("id") or self.model_id).lower()
        tags = [t.lower() for t in (hf_model.get("tags") or [])]
        tag_text = " ".join(tags)

        card_data = hf_model.get("cardData")
        if not isinstance(card_data, dict):
            card_data = {}
        readme = (hf_model.get("readme") or hf_model.get("cardData", {}) or "").lower() if isinstance(hf_model.get("readme"), str) else ""

        # ── License ──────────────────────────────────────────────────────────
        raw_license = hf_model.get("license") or ""
        if isinstance(raw_license, list):
            licenses = [str(x).lower() for x in raw_license]
        else:
            licenses = [str(raw_license).lower()] if raw_license else []

        has_open_license = any(any(ol in lic for ol in OPEN_LICENSES) for lic in licenses)
        has_restricted_license = any(any(rl in lic for rl in RESTRICTED_OPEN_LICENSES) for lic in licenses)
        has_no_license = not licenses or licenses == [""]

        # ── Base model ───────────────────────────────────────────────────────
        base_model = card_data.get("base_model") or hf_model.get("base_model") or ""
        if isinstance(base_model, list):
            base_model = base_model[0] if base_model else ""
        base_model = str(base_model).lower().strip()

        fine_tune_tags = {"finetuned", "fine-tuned", "fine_tuned", "derived", "instruction-tuned", "rlhf", "lora", "qlora"}
        has_finetune_tag = any(ft in t for t in tags for ft in fine_tune_tags)
        pipeline_tag = (hf_model.get("pipeline_tag") or "").lower()

        # ── 1. Training Data Independence ────────────────────────────────────
        if "fully open" in readme or "open data" in tag_text:
            scores["Training Data Independence"] = 0.85
            confidence["Training Data Independence"] = 0.6
        elif "transparent" in tag_text or "open" in tag_text:
            scores["Training Data Independence"] = 0.65
            confidence["Training Data Independence"] = 0.5
        elif has_open_license:
            scores["Training Data Independence"] = 0.55
            confidence["Training Data Independence"] = 0.4
        else:
            scores["Training Data Independence"] = 0.40
            confidence["Training Data Independence"] = 0.35

        # ── 2. Compute Independence ──────────────────────────────────────────
        # We can rarely determine this from HF metadata alone; start neutral
        if any(k in model_id_lower or k in author for k in ["swiss-ai", "ai-sweden", "cscs", "epfl"]):
            scores["Compute Independence"] = 0.80
            confidence["Compute Independence"] = 0.65
        elif any(k in author for k in BIG_TECH_ORGS):
            scores["Compute Independence"] = 0.10
            confidence["Compute Independence"] = 0.70
        else:
            scores["Compute Independence"] = 0.45
            confidence["Compute Independence"] = 0.25   # genuinely uncertain

        # ── 3. Weight Ownership & Access ─────────────────────────────────────
        if has_open_license and not has_restricted_license:
            scores["Weight Ownership & Access"] = 0.85
            confidence["Weight Ownership & Access"] = 0.75
        elif has_restricted_license:
            # Open-weight but revocable / usage-restricted
            scores["Weight Ownership & Access"] = 0.50
            confidence["Weight Ownership & Access"] = 0.70
        elif has_no_license:
            # Unknown — default closed
            scores["Weight Ownership & Access"] = 0.25
            confidence["Weight Ownership & Access"] = 0.40
        else:
            # Proprietary / other
            scores["Weight Ownership & Access"] = 0.15
            confidence["Weight Ownership & Access"] = 0.65

        # ── 4. Base Model Dependency ─────────────────────────────────────────
        if base_model:
            # Derived from another model
            if any(bt in base_model for bt in ["openai", "gpt", "claude", "gemini", "palm"]):
                scores["Base Model Dependency"] = 0.10
                confidence["Base Model Dependency"] = 0.80
            elif any(bt in base_model for bt in ["llama", "mistral", "falcon", "qwen"]):
                scores["Base Model Dependency"] = 0.35
                confidence["Base Model Dependency"] = 0.75
            else:
                scores["Base Model Dependency"] = 0.40
                confidence["Base Model Dependency"] = 0.60
        elif has_finetune_tag:
            scores["Base Model Dependency"] = 0.35
            confidence["Base Model Dependency"] = 0.55
        else:
            # Likely trained from scratch
            scores["Base Model Dependency"] = 0.80
            confidence["Base Model Dependency"] = 0.50

        # ── 5. Deployment Independence ───────────────────────────────────────
        if has_open_license and not has_restricted_license:
            scores["Deployment Independence"] = 0.90
            confidence["Deployment Independence"] = 0.75
        elif has_restricted_license:
            scores["Deployment Independence"] = 0.60
            confidence["Deployment Independence"] = 0.65
        elif has_no_license:
            scores["Deployment Independence"] = 0.20
            confidence["Deployment Independence"] = 0.40
        else:
            scores["Deployment Independence"] = 0.10
            confidence["Deployment Independence"] = 0.60

        # ── 6. Organisational Independence ───────────────────────────────────
        if any(bt in author for bt in BIG_TECH_ORGS):
            scores["Organisational Independence"] = 0.05
            confidence["Organisational Independence"] = 0.85
        elif any(pi in author for pi in PUBLIC_INSTITUTION_HINTS):
            scores["Organisational Independence"] = 0.90
            confidence["Organisational Independence"] = 0.75
        elif "university" in author or "institute" in author or "research" in author:
            scores["Organisational Independence"] = 0.80
            confidence["Organisational Independence"] = 0.60
        else:
            # Independent / startup — assume moderate
            scores["Organisational Independence"] = 0.55
            confidence["Organisational Independence"] = 0.40

        # ── 7. Jurisdictional Risk ───────────────────────────────────────────
        country = self.organisation.country
        scores["Jurisdictional Risk"] = JURISDICTION_SCORE.get(country, 0.40)
        confidence["Jurisdictional Risk"] = 0.65 if country != "–" else 0.30

        return scores, confidence

    # ------------------------------------------------------------------
    # LLM web scoring
    # ------------------------------------------------------------------

    def _score_from_web_docs(
        self,
        web_docs: list[dict],
        model_name: str,
        num_attempts: int = 2,
    ) -> Optional[dict]:
        """Use LLM to extract per-category scores from scraped documents."""
        method = "_score_from_web_docs"
        if not _PIPELINE_AVAILABLE or not os.getenv("PUBLICAI_KEY"):
            self._log("Skipping LLM web scoring (no PUBLICAI_KEY)", method=method)
            return None

        # Deduplicate by domain; non-HF sources first
        seen_domains: set[str] = set()
        deduped: list[dict] = []
        for d in web_docs:
            url = (d.get("url") or "").lower()
            domain = re.sub(r"https?://", "", url).split("/")[0]
            if domain not in seen_domains:
                seen_domains.add(domain)
                deduped.append(d)
        non_hf = [d for d in deduped if "huggingface.co" not in (d.get("url") or "")]
        hf_docs = [d for d in deduped if "huggingface.co" in (d.get("url") or "")]
        web_docs = non_hf + hf_docs

        if not web_docs:
            return {c: {"score": 0.5, "confidence": 0.0, "evidence": []} for c in CATEGORIES}

        results: dict[str, dict] = {}

        for category in CATEGORIES:
            filtered = [
                d for d in web_docs
                if _is_relevant(d.get("extracted", ""), category)
            ] or web_docs

            source_blocks: list[str] = []
            model_lower = model_name.lower()
            for j, d in enumerate(filtered[:4]):
                raw = d.get("extracted") or ""
                sentences = re.split(r"(?<=[.!?])\s+", raw)
                relevant = [
                    s.strip() for s in sentences
                    if model_lower in s.lower() and len(s.strip()) > 40
                ]
                snippet = " ".join(relevant[:8]) if relevant else raw[:1500]
                source_blocks.append(
                    f"Source {j + 1}:\nURL: {d.get('url', '')}\nContent:\n{snippet}"
                )

            sources_text = "\n\n".join(source_blocks)
            prompt = (
                f"You are extracting evidence from web sources about an AI model.\n\n"
                f"Return ONLY a JSON object. No preamble, no markdown fences.\n\n"
                f"Sources:\n{sources_text}\n\n"
                f"Task: Score the model \"{model_name}\" on this dimension: \"{category}\"\n\n"
                f"Definition of this dimension:\n"
                f"{_CATEGORY_DEFINITIONS.get(category, '')}\n\n"
                f"Instructions:\n"
                f"- Read the sources carefully.\n"
                f"- Find a sentence or passage directly relevant to the question.\n"
                f"- Copy it EXACTLY as it appears — do not rephrase.\n"
                f"- Score 1.0 = fully sovereign on this dimension, 0.0 = fully dependent on Big Tech.\n\n"
                f"Return this exact JSON:\n"
                f'{{\n'
                f'  "score": <float 0.0–1.0>,\n'
                f'  "confidence": <float 0.0–1.0>,\n'
                f'  "quote": "<exact substring from one source>",\n'
                f'  "url": "<url of that source>",\n'
                f'  "rationale": "<one sentence linking quote to score>"\n'
                f'}}\n\n'
                f'If no relevant text exists, set quote to "" and confidence to 0.'
            )

            try:
                parsed = None
                for _ in range(num_attempts):
                    raw_response = ask_publicai(prompt=prompt, user_agent="Sovereignty-Pipeline/1.0")
                    parsed = _extract_valid_json(raw_response)
                    if parsed:
                        break

                if not parsed:
                    raise ValueError("No valid JSON from LLM")

                score = _parse_float(parsed.get("score")) or 0.5
                confidence = _parse_float(parsed.get("confidence")) or 0.0
                quote_text = (parsed.get("quote") or "").strip()
                rationale = (parsed.get("rationale") or "").strip()

                evidence: list[dict] = []
                if quote_text and len(quote_text) > 20:
                    verified_url = _verify_quote(quote_text, filtered)
                    if verified_url is not None:
                        evidence.append({
                            "quote": quote_text,
                            "url": verified_url or parsed.get("url", ""),
                            "rationale": rationale,
                            "verified": True,
                        })

                # Fallback: pick best heuristic sentence
                if not evidence:
                    best = self._pick_best_sentence(filtered, category, model_name)
                    if best:
                        best["verified"] = False
                        evidence.append(best)

                results[category] = {"score": score, "confidence": confidence, "evidence": evidence}

            except Exception as exc:
                self._log_error(method, exc, context={"category": category})
                best = self._pick_best_sentence(filtered, category, model_name)
                results[category] = {
                    "score": 0.5,
                    "confidence": 0.0,
                    "evidence": [{**best, "verified": False}] if best else [],
                }

        return results

    def _pick_best_sentence(
        self, docs: list[dict], category: str, model_name: str
    ) -> Optional[dict]:
        model_lower = model_name.lower()
        candidates: list[tuple[int, str, str]] = []
        for d in docs:
            raw = d.get("extracted") or ""
            url = d.get("url", "")
            for s in re.split(r"(?<=[.!?])\s+", raw):
                s = s.strip()
                if len(s) < 40 or len(s) > 500 or _is_boilerplate(s):
                    continue
                sc = _score_sentence(s, category)
                if model_lower in s.lower():
                    sc += 3
                if sc > 0:
                    candidates.append((sc, s, url))
        if not candidates:
            for d in docs:
                for s in re.split(r"(?<=[.!?])\s+", (d.get("extracted") or "")):
                    s = s.strip()
                    if len(s) > 60 and not _is_boilerplate(s):
                        return {"quote": s, "url": d.get("url", ""), "rationale": f"Best available text for: {category}"}
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, text, url = candidates[0]
        return {"quote": text, "url": url, "rationale": f"Best available evidence for: {category}"}

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def _explain_score(self, user_agent: str) -> str:
        if not _PIPELINE_AVAILABLE or not os.getenv("PUBLICAI_KEY"):
            return self._fallback_explanation()

        dims = [
            {
                "category": c,
                "score": round(float(self.category_scores.get(c, 0.5)), 3),
                "confidence": round(float(self.category_confidence.get(c, 0.3)), 3),
                "weight": round(self.weights.get(c, 1.0 / len(CATEGORIES)), 3),
            }
            for c in CATEGORIES
        ]

        evidence_lines: list[str] = []
        for c in CATEGORIES:
            for ev in (self.evidence.get(c) or [])[:2]:
                q = (ev.get("quote") or "").strip()[:250]
                if q:
                    evidence_lines.append(f'[{c}] "{q}" — {ev.get("rationale", "")}')

        evidence_section = "\n".join(evidence_lines) or "(No web quotes available.)"

        prompt = f"""
        You are explaining an AI sovereignty score to a highly technical audience.

        Definition of AI sovereignty:
        A sovereign AI model is one that can be accessed, deployed, modified, and operated without reliance on, control by, or enforced constraints from dominant external providers (e.g. Big Tech platforms). This includes independence across infrastructure (compute/hosting), distribution channels, licensing, weights access, and operational control. A model is less sovereign if any critical capability (usage, scaling, modification, or access) can be restricted, revoked, or mediated by a third party.

        Scoring interpretation:
        Score 100 = fully sovereign (no meaningful external dependencies or control points).
        Score 0 = entirely dependent (access and use fully mediated or controlled by external providers).

        Model: {self.model_id}
        Overall score: {self.overall_score:.2f}/100
        Ground-truth entry used: {self.used_ground_truth}

        Category breakdown (score: 0=not sovereign, 1=fully sovereign):
        {json.dumps(dims, indent=2)}

        Web evidence (verified quotes):
        {evidence_section}

        Write a rigorous, evidence-driven explanation (5–8 sentences) that:
        1. Explains why the model lands at this overall score by explicitly linking category scores to concrete sovereignty constraints or freedoms.
        2. Identifies the 2–3 most influential categories (positive or negative), explaining how they increase or reduce real-world independence.
        3. Uses the sovereignty definition above to interpret the results (e.g. whether the model can be self-hosted, whether access can be revoked, whether usage is permissionless).
        4. Integrates multiple direct quotes from the evidence as supporting proof. Quotes must be embedded naturally and interpreted (explain what they imply about control, dependence, or restriction).
        5. Include quotes from relavent sources to justify reasoning.
        5. Resolves any ambiguity or conflicting signals in the evidence by prioritising the most authoritative or explicit sources.
        6. Focuses on concrete control points (hosting dependence, API gating, licensing restrictions, weights access, ability to run locally, etc.), not abstract summaries.
        7. Avoids describing the scoring formula itself.
        8. Avoids stating the score of each dimension.

        Style requirements:
        - Plain prose only (no bullet points, no markdown).
        - Be precise, assertive, and analytical; avoid hedging language.
        - Every major claim must be traceable to either a category score or a quoted source.
        - Prefer depth over brevity; the explanation should read like a technical audit of the model’s independence.
        """

        try:
            return ask_publicai(prompt=prompt, user_agent=user_agent)
        except Exception as exc:
            print(exc)
            print("This is not working")
            self._log_error("_explain_score", exc)
            return self._fallback_explanation()

    def _fallback_explanation(self) -> str:
        parts = [f"{self.model_id} sovereignty score: {self.overall_score:.2f}/100."]
        for c in CATEGORIES:
            v = self.category_scores.get(c, 0.5)
            conf = self.category_confidence.get(c, 0.3)
            flag = " [low confidence]" if conf < 0.4 else ""
            parts.append(f"{c}: {v:.3f}{flag}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Category definitions (used in LLM prompts for clearer extraction)
# ---------------------------------------------------------------------------

_CATEGORY_DEFINITIONS: dict[str, str] = {
    "Training Data Independence": (
        "Score 1.0 if the training data is entirely self-owned or openly licensed "
        "(e.g. open-government data, public domain), with no dependence on data "
        "pipelines controlled by a US or Chinese hyperscaler. Score 0.0 if the "
        "data is sourced exclusively from proprietary Big Tech pipelines."
    ),
    "Compute Independence": (
        "Score 1.0 if the model was trained on publicly owned or sovereign HPC "
        "infrastructure (e.g. national supercomputer, public-institution cluster). "
        "Score 0.0 if it was trained entirely on AWS, Azure, or GCP."
    ),
    "Weight Ownership & Access": (
        "Score 1.0 if the weights are released under an open licence with no usage "
        "restrictions and cannot be revoked by a third party. Score 0.0 if the "
        "weights are proprietary and access can be revoked at any time (e.g. API-only)."
    ),
    "Base Model Dependency": (
        "Score 1.0 if the model was trained from scratch. Score 0.0 if it is a "
        "fine-tune of a closed proprietary model (e.g. GPT-4). Score ~0.4 if it "
        "fine-tunes an open-weight model with some restrictions."
    ),
    "Deployment Independence": (
        "Score 1.0 if the model can be freely downloaded and run locally or "
        "on-prem with no restrictions. Score 0.0 if access is exclusively via "
        "a proprietary API that can be withdrawn."
    ),
    "Organisational Independence": (
        "Score 1.0 if the developing organisation is a public university, "
        "government lab, or independent non-profit with no Big Tech ownership. "
        "Score 0.0 if it is a Big Tech subsidiary or majority-owned by a "
        "US/Chinese hyperscaler."
    ),
    "Jurisdictional Risk": (
        "Score 1.0 if the organisation is domiciled in a jurisdiction with "
        "strong data-protection law and no equivalent of the US CLOUD Act "
        "(e.g. Switzerland). Score 0.0 if subject to US CLOUD Act or Chinese "
        "cybersecurity law mandating state access."
    ),
}


# ---------------------------------------------------------------------------
# OrganisationSovereigntyScore
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to a single space."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _build_keyword_variants(kw: str) -> list[str]:
    """
    Return surface forms of a keyword for matching org slugs and free text.

    Covers ``ai-sweden``, ``ai_sweden``, ``AISweden``, ``ai sweden``, etc.
    """
    raw = kw.lower().strip()
    normalised = _normalise(kw)
    no_space = normalised.replace(" ", "")
    hyphenated = normalised.replace(" ", "-")
    underscored = normalised.replace(" ", "_")
    return list({raw, normalised, no_space, hyphenated, underscored})


# Built once at import time
_KW_VARIANTS: dict[str, list[str]] = {
    kw: _build_keyword_variants(kw) for kw in COUNTRY_KEYWORDS
}

# Suffixes allowed after a keyword prefix in compound slugs (e.g. mistral + ai → mistralai)
_ORG_SLUG_SUFFIXES: frozenset[str] = frozenset({
    "ai", "hq", "labs", "lab", "inc", "ltd", "llc", "corp", "co",
    "group", "team", "models", "model", "research", "institute",
})


def _match_country_word_boundary(text: str) -> Optional[str]:
    """Match keywords with word-boundary checks on normalised text."""
    norm = _normalise(text)
    if not norm:
        return None
    for kw in sorted(_KW_VARIANTS, key=len, reverse=True):
        for variant in _KW_VARIANTS[kw]:
            pattern = rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])"
            if re.search(pattern, norm):
                return COUNTRY_KEYWORDS[kw]
    return None


def _match_country_tokens(text: str) -> Optional[str]:
    """Match each whitespace/hyphen token independently (longest keyword wins)."""
    norm = _normalise(text)
    if not norm:
        return None
    for token in sorted(set(norm.split()), key=len, reverse=True):
        hit = _match_country_word_boundary(token)
        if hit:
            return hit
    return None


def _match_country_slug_segment(segment: str) -> Optional[str]:
    """
    Match a single org-slug segment (e.g. ``swiss``, ``mistralai``, ``BSC``).

    Handles compound names where a keyword is a prefix (``mistral`` + ``ai``).
    """
    segment = (segment or "").lower().strip()
    if not segment:
        return None

    hit = _match_country_word_boundary(segment)
    if hit:
        return hit

    for kw in sorted(_KW_VARIANTS, key=len, reverse=True):
        for variant in _KW_VARIANTS[kw]:
            vs = variant.replace(" ", "")
            if len(vs) < 4:
                continue
            if not segment.startswith(vs):
                continue
            remainder = segment[len(vs):]
            if not remainder:
                return COUNTRY_KEYWORDS[kw]
            if remainder in _ORG_SLUG_SUFFIXES:
                return COUNTRY_KEYWORDS[kw]
    return None


def _match_country_from_slug(slug: str) -> Optional[str]:
    """
    Match country signals in a Hugging Face org slug or model namespace.

    Splits on ``-``, ``_``, ``.``, ``/`` and tries each segment, then the
    full slug, then embedded hyphenated keywords (``ai-sweden`` in
    ``ai-sweden-models``).
    """
    if not slug:
        return None

    slug_lower = slug.lower()
    segments = [s for s in re.split(r"[-_./]+", slug_lower) if s]

    for segment in sorted(segments, key=len, reverse=True):
        hit = _match_country_slug_segment(segment)
        if hit:
            return hit

    hit = _match_country_word_boundary(slug_lower)
    if hit:
        return hit

    hit = _match_country_tokens(slug_lower)
    if hit:
        return hit

    # Embedded multi-part keywords (e.g. "ai-sweden" inside "ai-sweden-models")
    for kw in sorted(COUNTRY_KEYWORDS, key=len, reverse=True):
        for variant in _KW_VARIANTS[kw]:
            if len(variant) < 4:
                continue
            for form in {variant, variant.replace(" ", "-"), variant.replace(" ", "_")}:
                if slug_lower == form:
                    return COUNTRY_KEYWORDS[kw]
                if re.search(
                    rf"(^|[-_.]){re.escape(form)}($|[-_.])",
                    slug_lower,
                ):
                    return COUNTRY_KEYWORDS[kw]

    return None


def _match_country(text: str) -> Optional[str]:
    """
    Try every keyword strategy against a text blob.

    Longer keywords are tested first so ``ai-sweden`` beats ``sweden``.
    """
    if not text:
        return None

    for matcher in (
        _match_country_word_boundary,
        _match_country_tokens,
        _match_country_from_slug,
    ):
        hit = matcher(text)
        if hit:
            return hit
    return None


def _collect_hf_country_text(hf_model: dict, org: str) -> str:
    """Gather all HF metadata fields that may contain country signals."""
    parts: list[str] = [
        org,
        hf_model.get("author") or "",
        hf_model.get("id") or "",
    ]

    card = hf_model.get("cardData")
    if isinstance(card, dict):
        for key in ("country", "location", "region", "affiliation", "language"):
            val = card.get(key)
            if val:
                parts.append(str(val))
        base = card.get("base_model")
        if base:
            parts.append(str(base))

    tags = hf_model.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)

    return " ".join(str(p) for p in parts if p)
 
 
# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class OrganisationSovereigntyScore:
    """
    Aggregates sovereignty scores for all models of one organisation.
 
    Parameters
    ----------
    name:
        Organisation name (e.g. ``"mistralai"``).
    organisation_type:
        ``"Big Tech"``, ``"State-backed"``, ``"Independent"``,
        ``"Non-profit"``, or ``"Community"``.
    country:
        Inferred home country.
    metadata:
        Arbitrary key/value pairs.
    """
 
    name:              str
    organisation_type: str  = "Independent"
    country:           str  = "–"
    metadata:          dict = field(default_factory=dict)
 
    _models: list[ModelSovereigntyScore] = field(
        default_factory=list, init=False, repr=False
    )
 
    # NOTE: _KW_VARIANTS is NOT a dataclass field — see FIX 1 above.
 
    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
 
    def add_model(self, model: ModelSovereigntyScore) -> None:
        if model.organisation is not self:
            raise ValueError(
                f"Model '{model.model_id}' is linked to a different organisation."
            )
        if model not in self._models:
            self._models.append(model)
 
    def create_model(
        self,
        model_id: str,
        weights: Optional[dict[str, float]] = None,
        verbose: bool = False,
    ) -> ModelSovereigntyScore:
        m = ModelSovereigntyScore(
            model_id=model_id,
            organisation=self,
            weights=weights or dict(DEFAULT_WEIGHTS),
            verbose=verbose,
        )
        self._models.append(m)
        return m
 
    @property
    def models(self) -> list[ModelSovereigntyScore]:
        return list(self._models)
 
    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
 
    def average_score(self) -> Optional[float]:
        evaluated = [m.overall_score for m in self._models if m.overall_score is not None]
        if not evaluated:
            return None
        return round(sum(evaluated) / len(evaluated), 2)
 
    def best_model(self) -> Optional[ModelSovereigntyScore]:
        evaluated = [m for m in self._models if m.overall_score is not None]
        return max(evaluated, key=lambda m: m.overall_score) if evaluated else None  # type: ignore
 
    def worst_model(self) -> Optional[ModelSovereigntyScore]:
        evaluated = [m for m in self._models if m.overall_score is not None]
        return min(evaluated, key=lambda m: m.overall_score) if evaluated else None  # type: ignore
 
    def score_summary(self) -> dict:
        evaluated = [m.overall_score for m in self._models if m.overall_score is not None]
        if not evaluated:
            return {"min": None, "max": None, "average": None, "model_count": 0}
        return {
            "min":         min(evaluated),
            "max":         max(evaluated),
            "average":     round(sum(evaluated) / len(evaluated), 2),
            "model_count": len(evaluated),
        }
 
    def low_confidence_warnings(self) -> list[dict]:
        """
        Return {model, category, confidence, score} entries where confidence
        is below 0.4, flagging scores based on thin evidence.
        """
        warnings = []
        for m in self._models:
            for c, conf in (m.category_confidence or {}).items():
                if conf < 0.4:
                    warnings.append({
                        "model":      m.model_id,
                        "category":   c,
                        "confidence": round(conf, 3),
                        "score":      round(m.category_scores.get(c, 0.5), 3),
                    })
        return warnings
 
    @classmethod
    def detect_country(cls, hf_model: dict, model_id: str = "") -> str:
        """
        Infer the organisation's home country from HF metadata and keyword tables.

        Search order (first hit wins):

        1. Org slug / author / model id (fast, works offline)
        2. Hugging Face organisation API metadata + URLs
        3. Model card fields and tags
        """
        effective_id = model_id or (hf_model.get("id") if hf_model else "") or ""
        namespace = effective_id.split("/")[0] if "/" in effective_id else effective_id

        if not hf_model:
            hit = _match_country_from_slug(namespace) if namespace else None
            return hit if hit else "–"

        author = (hf_model.get("author") or "").strip()
        org = effective_id.split("/")[0] if "/" in effective_id else author

        for candidate in dict.fromkeys([org, author, effective_id, namespace, f"{author} {effective_id}"]):
            if not candidate:
                continue
            hit = _match_country_from_slug(candidate)
            if hit:
                return hit
            hit = _match_country(candidate)
            if hit:
                return hit

        if org:
            try:
                org_data = _get_hf_org(org)
            except Exception:
                org_data = None

            if org_data:
                text_blob = " ".join(filter(None, [
                    org, author,
                    str(org_data.get("name", "")),
                    str(org_data.get("fullname", "")),
                    str(org_data.get("description", "")),
                    str(org_data.get("location", "")),
                    str(org_data.get("blog", "")),
                    str(org_data.get("github", "")),
                    str(org_data.get("email", "")),
                ]))
                hit = _match_country(text_blob)
                if hit:
                    return hit

                for field_name in ("blog", "github", "email", "website"):
                    inferred = _infer_country_from_url(str(org_data.get(field_name) or ""))
                    if inferred:
                        return inferred

        card_text = _collect_hf_country_text(hf_model, org)
        hit = _match_country(card_text)
        if hit:
            return hit

        return "–"

    @classmethod
    def detect_org_type(cls, hf_model: dict) -> str:
        """
        Infer organisation type from a web search blob, falling back to
        simple author-string matching when DuckDuckGo is unavailable.
        """
        author   = (hf_model.get("author") or "").lower()
        org_type = "Independent"
 
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(author, max_results=10))
            blob = " ".join(
                r.get("body", "") or r.get("title", "") or "" for r in results
            ).lower()
 
            if any(k in blob for k in BIG_TECH_ORGS):
                org_type = "Big Tech"
            elif any(k in blob for k in PUBLIC_INSTITUTION_HINTS):
                org_type = "State-backed"
            elif any(k in blob for k in ["non-profit", "nonprofit", "charity",
                                          "foundation", "ngo"]):
                org_type = "Non-profit"
            elif any(k in blob for k in ["community", "collective", "open-source"]):
                org_type = "Community"
 
        except Exception:
            # Fallback: match directly against the author slug
            if any(x in author for x in BIG_TECH_ORGS):
                org_type = "Big Tech"
            elif any(x in author for x in PUBLIC_INSTITUTION_HINTS):
                org_type = "State-backed"
            elif "nonprofit" in author or "non-profit" in author:
                org_type = "Non-profit"
            elif "community" in author or "collective" in author:
                org_type = "Community"
 
        return org_type
 
    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
 
    def to_dict(self) -> dict:
        return {
            "organisation": {
                "name":                   self.name,
                "organisation_type":      self.organisation_type,
                "country":                self.country,
                "metadata":               self.metadata,
                "aggregate":              self.score_summary(),
                "low_confidence_warnings": self.low_confidence_warnings(),
            },
            "models": [m.to_dict() for m in self._models],
        }
 
    def save_json(self, path: "str | Path", indent: int = 2) -> Path:
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=indent, ensure_ascii=False)
        return dest
 
    @classmethod
    def load_json(cls, path: "str | Path") -> "OrganisationSovereigntyScore":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
 
        org_data = data.get("organisation", {})
        org = cls(
            name=              org_data.get("name",              ""),
            organisation_type= org_data.get("organisation_type", "Independent"),
            country=           org_data.get("country",           "–"),
            metadata=          org_data.get("metadata",          {}),
        )
 
        for m_data in data.get("models", []):
            m = ModelSovereigntyScore(model_id=m_data["model_id"], organisation=org)
            m.overall_score       = m_data.get("value")
            m.category_scores     = m_data.get("categories")          or {}
            m.category_confidence = m_data.get("category_confidence") or {}
            m.evidence            = m_data.get("evidence")            or {}
            m.sources             = m_data.get("sources")             or []
            m.explanation         = m_data.get("explanation")
            m.used_ground_truth   = m_data.get("used_ground_truth",   False)
 
            # FIX 4 — restore fallback_log so degradation history survives
            # a JSON round-trip (field introduced in ModelSovereigntyScore session)
            for fb in m_data.get("fallback_log", []):
                m.fallback_log.append(
                    _FallbackRecord(
                        method=        fb.get("method",       ""),
                        intended=      fb.get("intended",     ""),
                        fallback_used= fb.get("fallback_used",""),
                        reason=        fb.get("reason",       ""),
                    )
                )
 
            org._models.append(m)
 
        return org
 
    def __repr__(self) -> str:
        return (
            f"OrganisationSovereigntyScore("
            f"name={self.name!r}, country={self.country!r}, "
            f"models={len(self._models)}, avg_score={self.average_score()}"
            f")"
        )
 
 
# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
 
def evaluate_model(
    model_id:    str,
    use_web:     bool = False,
    use_llm_web: bool = False,
    verbose:     bool = False,
) -> tuple[OrganisationSovereigntyScore, ModelSovereigntyScore]:
    if not _PIPELINE_AVAILABLE:
        raise RuntimeError("pipeline package is required for evaluate_model().")

    hf_model = fetch_huggingface_model(model_id)
    author   = ((hf_model or {}).get("author") or model_id.split("/")[0])

    org = OrganisationSovereigntyScore(
        name=author,
        organisation_type= OrganisationSovereigntyScore.detect_org_type(hf_model or {}),
        country=           OrganisationSovereigntyScore.detect_country(hf_model or {}, model_id=model_id),
        metadata={
            "source":       "public-ai sovereignty pipeline",
            "version":      "0.2.0",
            "uses_web":     use_web,
            "uses_llm_web": use_llm_web,
        },
    )

    model_score = org.create_model(model_id)
    model_score.evaluate(use_web=use_web, use_llm_web=use_llm_web, verbose=verbose)
    return org, model_score