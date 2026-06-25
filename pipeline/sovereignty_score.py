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
2. Compute Independence
3. Weight Ownership & Access
4. Base Model Dependency
5. Deployment Independence
6. Organisational Independence
7. Jurisdictional Risk

Score: 0–100. Higher = more sovereign (less dependent on Big Tech).

Improvement strategies implemented
-----------------------------------
1. Ground-truth fuzzy matching     — org-level inherit + family/version normalisation
2. Richer HF metadata extraction   — README compute/data mining, datasets field,
                                     siblings (GGUF/ONNX), Spaces, param count
3. Dimension-specific web queries  — 7 targeted searches, one per dimension,
                                     confidence-gated (only fired when HF conf < threshold)
4. Confidence-gated escalation     — per-dimension web escalation; avoids paying
                                     for web calls on dimensions already well-scored
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
# Confidence threshold for Strategy 4: confidence-gated web escalation.
# Dimensions whose HF-heuristic confidence is below this value will trigger
# a targeted web search rather than keeping the uncertain HF score.
# ---------------------------------------------------------------------------

WEB_ESCALATION_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Dimensions that are stable *across all models* from the same organisation.
# Used by Strategy 1 (org-level inherit from ground truth).
# ---------------------------------------------------------------------------

ORG_STABLE_DIMENSIONS: set[str] = {
    "Jurisdictional Risk",
    "Organisational Independence",
    "Compute Independence",      # usually the same infra per org
}

# ---------------------------------------------------------------------------
# Ground-truth table
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, dict[str, float]] = {
    # ── Big Tech / API-only ──────────────────────────────────────────────────
    "openai/gpt-4": {
        "Training Data Independence":  0.05,
        "Compute Independence":        0.05,
        "Weight Ownership & Access":   0.05,
        "Base Model Dependency":       0.90,
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
    # ── Meta open-weight ─────────────────────────────────────────────────────
    "meta-llama/llama-3-70b-instruct": {
        "Training Data Independence":  0.30,
        "Compute Independence":        0.10,
        "Weight Ownership & Access":   0.45,
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
    # ── Mistral (France) ─────────────────────────────────────────────────────
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
    # ── EleutherAI / Pythia ──────────────────────────────────────────────────
    "eleutherai/pythia-12b": {
        "Training Data Independence":  0.75,
        "Compute Independence":        0.55,
        "Weight Ownership & Access":   0.90,
        "Base Model Dependency":       0.90,
        "Deployment Independence":     0.95,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.30,
    },
    # ── Falcon / TII (UAE) ───────────────────────────────────────────────────
    "tiiuae/falcon-40b": {
        "Training Data Independence":  0.60,
        "Compute Independence":        0.65,
        "Weight Ownership & Access":   0.75,
        "Base Model Dependency":       0.85,
        "Deployment Independence":     0.85,
        "Organisational Independence": 0.70,
        "Jurisdictional Risk":         0.60,
    },
    # ── Swiss-AI / EPFL ──────────────────────────────────────────────────────
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

GROUND_TRUTH = {k.lower(): v for k, v in GROUND_TRUTH.items()}

# ---------------------------------------------------------------------------
# Strategy 1 helpers: org-level index and version-normalisation regex
# ---------------------------------------------------------------------------

# Pre-built: maps org slug → list of GT entries for that org.
# Built once at import time from GROUND_TRUTH.
_GT_BY_ORG: dict[str, list[dict[str, float]]] = {}
for _gt_key, _gt_scores in GROUND_TRUTH.items():
    _org_slug = _gt_key.split("/")[0] if "/" in _gt_key else _gt_key
    _GT_BY_ORG.setdefault(_org_slug, []).append(_gt_scores)

# Suffixes to strip when normalising model IDs for family matching.
_VERSION_STRIP_RE = re.compile(
    r"[-_](?:"
    r"v\d+(?:[._]\d+)*"          # v0.1, v2, v0.1.0
    r"|instruct|chat|hf"         # common suffixes
    r"|it|rlhf|sft|dpo"
    r"|\d+b|\d+x\d+b"            # size suffixes: 7b, 8x7b
    r"|preview|rc\d*|alpha|beta"
    r")",
    re.IGNORECASE,
)


def _normalise_model_id(model_id: str) -> str:
    """Strip version/size/instruction suffixes for family matching."""
    low = model_id.lower()
    return _VERSION_STRIP_RE.sub("", low).rstrip("-_")


def _gt_family_lookup(model_id: str) -> Optional[dict[str, float]]:
    """
    Strategy 1a — exact or family match against GROUND_TRUTH.

    Returns (scores_dict, confidence_multiplier).
    Exact match → 1.0. Family/normalised match → 0.85.
    Returns None if no match.
    """
    low = model_id.lower()
    if low in GROUND_TRUTH:
        return GROUND_TRUTH[low], 1.0

    normalised = _normalise_model_id(low)
    for gt_key in GROUND_TRUTH:
        if _normalise_model_id(gt_key) == normalised:
            return GROUND_TRUTH[gt_key], 0.85

    return None, None


def _gt_org_inherit(model_id: str) -> Optional[dict[str, float]]:
    """
    Strategy 1b — org-level inherit for ORG_STABLE_DIMENSIONS.

    If no family match exists but the org is in the ground truth, return a
    partial dict covering only ORG_STABLE_DIMENSIONS (averaged across all
    GT entries for that org).  Returns None if org is unknown.
    """
    org = model_id.lower().split("/")[0] if "/" in model_id else ""
    entries = _GT_BY_ORG.get(org)
    if not entries:
        return None
    partial: dict[str, float] = {}
    for dim in ORG_STABLE_DIMENSIONS:
        vals = [e[dim] for e in entries if dim in e]
        if vals:
            partial[dim] = sum(vals) / len(vals)
    return partial if partial else None


# ---------------------------------------------------------------------------
# Known open / permissive licences
# ---------------------------------------------------------------------------

OPEN_LICENSES: set[str] = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause",
    "cc-by-4.0", "cc-by-sa-4.0",
    "openrail", "openrail++", "bigscience-openrail-m",
    "afl-3.0", "artistic-2.0",
    "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0",
    "llama2", "llama3",
    "cc0-1.0", "wtfpl", "unlicense",
}

RESTRICTED_OPEN_LICENSES: set[str] = {
    "llama2", "llama3", "cc-by-nc-4.0", "cc-by-nc-sa-4.0",
    "gemma", "microsoft-research-license",
}

# ---------------------------------------------------------------------------
# Jurisdiction tiers
# ---------------------------------------------------------------------------

JURISDICTION_SCORE: dict[str, float] = {
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
    "United Kingdom":   0.55,
    "Canada":           0.50,
    "Australia":        0.48,
    "New Zealand":      0.48,
    "United States":    0.15,
    "China":            0.10,
    "United Arab Emirates": 0.55,
    "Saudi Arabia":     0.45,
    "Israel":           0.60,
    "Japan":            0.65,
    "South Korea":      0.62,
    "Singapore":        0.60,
    "India":            0.55,
    "Russia":           0.10,
    "Taiwan":           0.65,
    "–":                0.40,
}

# ---------------------------------------------------------------------------
# Country detection
# ---------------------------------------------------------------------------

COUNTRY_KEYWORDS: dict[str, str] = {
    "swiss-ai": "Switzerland", "swiss_ai": "Switzerland",
    "swiss": "Switzerland", "switzerland": "Switzerland",
    "epfl": "Switzerland", "eth zurich": "Switzerland",
    "eth zürich": "Switzerland", "cscs": "Switzerland",
    "swisstxt": "Switzerland",
    "ai-sweden": "Sweden", "ai sweden": "Sweden", "ai-sweden-models": "Sweden",
    "sweden": "Sweden",
    "finland": "Finland", "norway": "Norway", "denmark": "Denmark",
    "iceland": "Iceland",
    "mistral": "France", "mistralai": "France", "lighton": "France", "france": "France",
    "huggingface": "France",
    "aleph alpha": "Germany", "alephalpha": "Germany", "germany": "Germany",
    "utter-project": "European Union", "eu": "European Union",
    "european union": "European Union",
    "bsc-lt": "Spain", "spain": "Spain",
    "italy": "Italy", "portugal": "Portugal",
    "belgium": "Belgium", "netherlands": "Netherlands",
    "stability": "United Kingdom", "stabilityai": "United Kingdom",
    "deepmind": "United Kingdom", "uk": "United Kingdom",
    "united kingdom": "United Kingdom", "britain": "United Kingdom",
    "ucl": "United Kingdom", "oxford": "United Kingdom",
    "cambridge": "United Kingdom",
    "openai": "United States", "anthropic": "United States",
    "meta": "United States", "google": "United States",
    "microsoft": "United States", "amazon": "United States",
    "aws": "United States", "nvidia": "United States",
    "allenai": "United States", "eleutherai": "United States",
    "xai": "United States", "x.ai": "United States",
    "cohere": "United States",
    "deepseek": "China", "qwen": "China", "alibaba": "China",
    "baidu": "China", "ernie": "China", "pangu": "China",
    "huawei": "China", "zhipu": "China", "chatglm": "China",
    "giga-llm": "China", "01-ai": "China", "yi-": "China",
    "moonshot": "China", "minimax": "China",
    "falcon": "United Arab Emirates", "tiiuae": "United Arab Emirates",
    "tii": "United Arab Emirates",
    "technology innovation institute": "United Arab Emirates",
    "mbzuai": "United Arab Emirates",
    "allam": "Saudi Arabia", "sdaia": "Saudi Arabia",
    "naver": "South Korea", "hyperclova": "South Korea",
    "kakao": "South Korea", "skt": "South Korea",
    "sarvam": "India", "ai4bharat": "India", "india": "India",
    "riken": "Japan", "fugaku": "Japan", "jaist": "Japan",
    "aisingapore": "Singapore", "ai singapore": "Singapore",
    "singapore": "Singapore", "sea-lion": "Singapore", "sea lion": "Singapore",
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

CLOUD_PROVIDERS: set[str] = {
    "azure", "aws", "amazon", "google cloud", "gcp",
    "lambda labs", "coreweave", "oracle cloud",
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
# Strategy 2: additional README/metadata signals
# ---------------------------------------------------------------------------

# File extensions that indicate the model can be run locally without the
# original framework — strong signal for Deployment Independence.
LOCAL_RUN_EXTENSIONS: set[str] = {
    "gguf", "ggml", "onnx", "bin", "pt", "safetensors",
}

# Sibling filename patterns that strongly imply local runnability.
LOCAL_RUN_PATTERNS: list[str] = [
    r"\.gguf$", r"\.ggml$", r"\.onnx$",
    r"quantiz", r"q4_", r"q8_", r"llama\.cpp",
]

# Known open HF dataset orgs — used to score Training Data Independence
# when the card's `datasets:` field references them.
OPEN_DATASET_ORGS: set[str] = {
    "allenai", "huggingface", "EleutherAI", "bigscience",
    "cc100", "oscar-corpus", "wikipedia", "common_voice",
    "openassistant", "lmsys", "databricks", "togethercomputer",
    "tiiuae",                  # Falcon's RefinedWeb is open
}

PROPRIETARY_DATASET_HINTS: list[str] = [
    "proprietary", "internal", "private", "closed", "undisclosed",
    "not disclosed", "confidential",
]

# ---------------------------------------------------------------------------
# Strategy 3: per-dimension web query templates
# ---------------------------------------------------------------------------

DIMENSION_QUERIES: dict[str, list[str]] = {
    "Training Data Independence": [
        '"{model}" training data dataset sources',
        '"{org}" training corpus open data',
    ],
    "Compute Independence": [
        '"{model}" trained on supercomputer HPC cluster',
        '"{model}" trained AWS Azure GCP cloud compute',
    ],
    "Weight Ownership & Access": [
        '"{model}" model weights license access download',
        '"{model}" open weights revoke terms',
    ],
    "Base Model Dependency": [
        '"{model}" base model fine-tune pretrained from',
        '"{model}" trained from scratch pretraining',
    ],
    "Deployment Independence": [
        '"{model}" run locally self-host on-prem deploy',
        '"{model}" API only hosted inference',
    ],
    "Organisational Independence": [
        '"{org}" nonprofit university venture capital funding',
        '"{org}" organisation type public institution',
    ],
    "Jurisdictional Risk": [
        '"{org}" headquarters country incorporated domicile',
        '"{org}" legal entity jurisdiction',
    ],
}

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
        "and not dependent on proprietary or restricted datasets."
    ),
    "Compute Independence": (
        "Assesses whether the model was trained and can be run using independent, sovereign, "
        "or publicly accessible compute infrastructure."
    ),
    "Weight Ownership & Access": (
        "Indicates whether the model weights are owned and controlled by an independent entity "
        "and released under an open licence."
    ),
    "Base Model Dependency": (
        "Evaluates the extent to which the model depends on proprietary, closed, or big tech "
        "base models vs being trained from scratch."
    ),
    "Deployment Independence": (
        "Reflects the ability to deploy and operate the model in any environment, including "
        "on-premises, without API-only or licensing restrictions."
    ),
    "Organisational Independence": (
        "Scores the independence of the developing organisation from big tech or foreign "
        "government control."
    ),
    "Jurisdictional Risk": (
        "Assesses the legal and regulatory exposure of the organisation to high-risk "
        "jurisdictions such as those covered by the US CLOUD Act."
    ),
}

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
# Strategy 2: README/metadata mining helpers
# ---------------------------------------------------------------------------

def _extract_param_count(hf_model: dict) -> Optional[int]:
    """
    Try to extract parameter count (as int) from HF metadata.
    Sources: safetensors metadata, tags like '7b', model_id.
    """
    # 1. Safetensors metadata
    st = hf_model.get("safetensors") or {}
    total = st.get("total")
    if isinstance(total, int) and total > 0:
        return total

    # 2. Tags like '7b', '13b', '70b', '0.5b'
    tags = [t.lower() for t in (hf_model.get("tags") or [])]
    model_id = (hf_model.get("id") or "").lower()
    for candidate in tags + [model_id]:
        m = re.search(r"(\d+(?:\.\d+)?)b(?:\b|_|-)", candidate)
        if m:
            try:
                return int(float(m.group(1)) * 1_000_000_000)
            except ValueError:
                pass
    return None


def _has_local_run_siblings(hf_model: dict) -> bool:
    """
    Return True if the model has sibling files (GGUF, ONNX, etc.) that imply
    it can be run locally without the original framework.
    """
    siblings = hf_model.get("siblings") or []
    for s in siblings:
        fname = (s.get("rfilename") or "").lower()
        if any(re.search(pat, fname) for pat in LOCAL_RUN_PATTERNS):
            return True
    return False


def _has_spaces(hf_model: dict) -> bool:
    """Return True if the model card links to any Hugging Face Spaces."""
    spaces = hf_model.get("spaces") or []
    return bool(spaces)


def _score_datasets_field(card_data: dict) -> Optional[float]:
    """
    Inspect the `datasets:` field of the model card.
    Returns a float [0,1] for Training Data Independence, or None if absent.
    """
    datasets = card_data.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    if not datasets:
        return None

    open_count = 0
    closed_count = 0
    for ds in datasets:
        ds_lower = str(ds).lower()
        if any(hint in ds_lower for hint in PROPRIETARY_DATASET_HINTS):
            closed_count += 1
        elif any(org.lower() in ds_lower for org in OPEN_DATASET_ORGS):
            open_count += 1
        else:
            # Unknown: treat as mildly open (benefit of the doubt)
            open_count += 0.5
    total = open_count + closed_count
    if total == 0:
        return None
    return min(1.0, open_count / total)


def _score_readme_compute(readme: str) -> Optional[tuple[float, float]]:
    """
    Scan the README for compute hints.
    Returns (score, confidence) or None if no signal found.
    """
    if not readme:
        return None
    t = readme.lower()
    sovereign_hits = sum(1 for h in SOVEREIGN_COMPUTE_HINTS if h in t)
    cloud_hits = sum(1 for h in CLOUD_PROVIDERS if h in t)

    if sovereign_hits == 0 and cloud_hits == 0:
        return None
    if sovereign_hits > 0 and cloud_hits == 0:
        return (min(1.0, 0.70 + 0.05 * sovereign_hits), min(0.85, 0.60 + 0.05 * sovereign_hits))
    if cloud_hits > 0 and sovereign_hits == 0:
        return (max(0.05, 0.25 - 0.05 * cloud_hits), min(0.85, 0.65 + 0.05 * cloud_hits))
    # Mixed: lean toward cloud (more conservative / lower sovereignty)
    return (0.35, 0.50)


def _score_readme_data(readme: str) -> Optional[tuple[float, float]]:
    """
    Scan the README for training-data transparency signals.
    Returns (score, confidence) or None.
    """
    if not readme:
        return None
    t = readme.lower()
    positive = [
        "open data", "public domain", "creative commons",
        "openly licensed", "cc-by", "cc0",
        "data card", "data sheet", "data transparency",
    ]
    negative = PROPRIETARY_DATASET_HINTS
    pos = sum(1 for p in positive if p in t)
    neg = sum(1 for n in negative if n in t)
    if pos == 0 and neg == 0:
        return None
    if pos > neg:
        return (min(0.90, 0.55 + 0.08 * pos), min(0.75, 0.45 + 0.06 * pos))
    if neg > pos:
        return (max(0.10, 0.45 - 0.10 * neg), min(0.75, 0.50 + 0.05 * neg))
    return (0.50, 0.40)


# ---------------------------------------------------------------------------
# ModelSovereigntyScore
# ---------------------------------------------------------------------------

@dataclass
class ModelSovereigntyScore:
    """
    Sovereignty score for a single AI model.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier.
    organisation : OrganisationSovereigntyScore
        Parent organisation object.
    weights : dict
        Per-category weights (must sum to 1).
    verbose : bool
        Print coloured progress to the terminal.
    web_escalation_threshold : float
        Dimensions with HF-heuristic confidence below this value will be
        escalated to targeted web search (Strategy 4).
    """

    model_id: str
    organisation: "OrganisationSovereigntyScore"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    verbose: bool = False
    web_escalation_threshold: float = WEB_ESCALATION_THRESHOLD

    overall_score: Optional[float] = field(default=None, init=False)
    category_scores: dict[str, float] = field(default_factory=dict, init=False)
    category_confidence: dict[str, float] = field(default_factory=dict, init=False)
    evidence: dict[str, list[dict]] = field(default_factory=dict, init=False)
    sources: list[str] = field(default_factory=list, init=False)
    explanation: Optional[str] = field(default=None, init=False)
    used_ground_truth: bool = field(default=False, init=False)
    gt_match_type: Optional[str] = field(default=None, init=False)  # 'exact'|'family'|'org_inherit'

    _hf_model: Optional[dict] = field(default=None, init=False, repr=False)

    _LOG_RESET = "\033[0m"
    _LOG_RED   = "\033[91m"
    _LOG_BLUE  = "\033[94m"
    _LOG_WHITE = "\033[97m"

    def _log(self, message: str, *, method: str) -> None:
        if self.verbose:
            print(f"{self._LOG_BLUE}[INFO] ModelSovereigntyScore.{method}: {message}{self._LOG_RESET}", flush=True)

    def _log_general(self, message: str, *, method: str) -> None:
        if self.verbose:
            print(f"{self._LOG_WHITE}ModelSovereigntyScore.{method}: {message}{self._LOG_RESET}", flush=True)

    def _log_error(self, method: str, exc: BaseException, *, context: Optional[dict] = None) -> None:
        if not self.verbose:
            return
        lines = [f"[ERROR] ModelSovereigntyScore.{method}", f"Model: {self.model_id}"]
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

        # NOTE on fetch strategy (bugfix — previously this caused up to 3x
        # redundant web collection):
        #
        # When use_llm_web=True, _compute_and_store() runs its own targeted,
        # per-dimension fetches via _fetch_dimension_docs() (Strategy 3), and
        # those targeted fetches are almost always non-empty — meaning the
        # generic fetch_web_evidence() call below was being made up front,
        # then thrown away unused, while two more fetch passes happened
        # downstream (the per-dimension loop, and a separate jurisdiction
        # fetch). We now only do the generic, single-query fetch here when
        # use_llm_web is False (i.e. when the cheap heuristic web-adjustment
        # path in _compute_and_store will actually consume web_docs). When
        # use_llm_web is True, web_docs stays empty and all web collection is
        # delegated to the single confidence-gated pass inside
        # _compute_and_store, which fetches each under-confident dimension
        # exactly once.
        web_docs: list[dict] = []
        if use_web and not use_llm_web:
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
        try:
            self.explanation = self._explain_score(user_agent=user_agent)
        except Exception as exc:
            self._log_error("generate_explanation", exc)
            self.explanation = self._fallback_explanation()
        return self.explanation

    def to_dict(self) -> dict:
        return {
            "model_id":             self.model_id,
            "author":               self.organisation.name,
            "metric_name":          "sovereignty_score",
            "metric_type":          "custom",
            "value":                self.overall_score,
            "categories":           self.category_scores,
            "category_confidence":  self.category_confidence,
            "evidence":             self.evidence,
            "sources":              self.sources,
            "explanation":          self.explanation,
            "used_ground_truth":    self.used_ground_truth,
            "gt_match_type":        self.gt_match_type,
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
        method = "_compute_and_store"
        self._log_general(f"Computing sovereignty score for '{self.model_id}'", method=method)

        # ── Strategy 1a: exact or family ground-truth match ──────────────────
        gt_scores, gt_conf_mult = _gt_family_lookup(self.model_id)
        if gt_scores is not None:
            match_type = "exact" if gt_conf_mult == 1.0 else "family"
            self._log(f"Ground-truth {match_type} match for '{self.model_id}'", method=method)
            self.category_scores     = {c: gt_scores.get(c, 0.5) for c in CATEGORIES}
            self.category_confidence = {c: gt_conf_mult           for c in CATEGORIES}
            self.evidence            = {}
            self.used_ground_truth   = True
            self.gt_match_type       = match_type
            self.overall_score       = self._weighted_total(self.category_scores)
            return

        # ── Strategy 2: richer HF metadata heuristics ────────────────────────
        hf_scores, hf_confidence = self._score_from_huggingface(hf_model or {})

        # ── Strategy 1b: org-level inherit for stable dimensions ─────────────
        org_inherit = _gt_org_inherit(self.model_id)
        if org_inherit:
            self._log(
                f"Applying org-level ground-truth inherit for: {list(org_inherit)}", method=method
            )
            for dim, val in org_inherit.items():
                hf_scores[dim]     = val
                hf_confidence[dim] = 0.80   # high but not ground-truth certain
            if not self.gt_match_type:
                self.gt_match_type = "org_inherit"

        # ── Strategy 4: confidence-gated web escalation ───────────────────────
        # Identify which dimensions are under-confident and need web support.
        dims_needing_web: set[str] = {
            c for c in CATEGORIES
            if hf_confidence.get(c, 0.3) < self.web_escalation_threshold
        }

        evidence_map: dict[str, list[dict]] = {}
        final_scores     = dict(hf_scores)
        final_confidence = dict(hf_confidence)

        # ── Strategy 3 + 4: dimension-specific targeted web queries ──────────
        if use_llm_web and dims_needing_web:
            self._log(
                f"Targeted web escalation for: {sorted(dims_needing_web)}", method=method
            )
            model_name = self.model_id.split("/")[-1]
            org_name   = self.model_id.split("/")[0] if "/" in self.model_id else ""

            for category in dims_needing_web:
                targeted_docs = self._fetch_dimension_docs(
                    category=category,
                    model_name=model_name,
                    org_name=org_name,
                )
                if targeted_docs:
                    self.sources = sorted({
                        *self.sources,
                        *(d.get("url") for d in targeted_docs if d.get("url")),
                    })

                dim_result = self._score_dimension_with_llm(
                    category=category,
                    docs=targeted_docs,
                    model_name=model_name,
                )
                if dim_result:
                    web_score = dim_result.get("score", 0.5)
                    web_conf  = dim_result.get("confidence", 0.0)
                    # Blend: up to 65 % weight to web when its confidence is high
                    alpha = min(0.65, web_conf)
                    final_scores[category] = (1 - alpha) * hf_scores[category] + alpha * web_score
                    final_confidence[category] = max(hf_confidence[category], web_conf)
                    evidence_map[category]      = dim_result.get("evidence", [])

        elif use_llm_web and web_docs:
            # Fallback: use the generic web docs if targeted fetch not available
            self._log("Applying heuristic web adjustments from generic docs", method=method)
            text = " ".join(d.get("extracted", "") for d in web_docs).lower()
            if any(k in text for k in SOVEREIGN_COMPUTE_HINTS):
                final_scores["Compute Independence"] = min(
                    1.0, final_scores["Compute Independence"] + 0.15
                )
            if any(k in text for k in CLOUD_PROVIDERS):
                final_scores["Compute Independence"] = max(
                    0.0, final_scores["Compute Independence"] - 0.20
                )

        # ── Strategy 4: targeted jurisdiction web query when country unknown ──
        # This is a fallback ONLY. If the per-dimension loop above already
        # fetched and scored "Jurisdictional Risk" (i.e. use_llm_web=True and
        # it was in dims_needing_web), we must not fetch for it again here —
        # that was the second source of the 3x-fetch bug. We only land in
        # this branch when jurisdiction was NOT already resolved by the loop.
        jurisdiction_already_handled = (
            use_llm_web and "Jurisdictional Risk" in dims_needing_web
        )
        if (
            self.organisation.country == "–"
            and not jurisdiction_already_handled
        ):
            self._log("Firing jurisdiction-specific web query (country unknown)", method=method)
            jur_score = self._fetch_jurisdiction_score(
                model_id=self.model_id,
                org_name=self.model_id.split("/")[0] if "/" in self.model_id else "",
            )
            if jur_score is not None:
                final_scores["Jurisdictional Risk"]     = jur_score
                final_confidence["Jurisdictional Risk"] = 0.55

        self.category_scores     = final_scores
        self.category_confidence = final_confidence
        self.evidence            = evidence_map
        self.overall_score       = self._weighted_total(final_scores)
        self._log(f"Overall sovereignty score: {self.overall_score}", method=method)

    def _weighted_total(self, scores: dict[str, float]) -> float:
        total = sum(
            scores.get(c, 0.5) * self.weights.get(c, 1.0 / len(CATEGORIES))
            for c in CATEGORIES
        )
        return round(total * 100, 2)

    # ------------------------------------------------------------------
    # Strategy 2: richer HF metadata heuristics
    # ------------------------------------------------------------------

    def _score_from_huggingface(
        self, hf_model: dict
    ) -> tuple[dict[str, float], dict[str, float]]:
        scores:     dict[str, float] = {c: 0.5 for c in CATEGORIES}
        confidence: dict[str, float] = {c: 0.3 for c in CATEGORIES}

        # ── Bugfix: Big-Tech-with-no-HF-card detector ───────────────────────
        # Models like GPT-5, Gemini, and Claude have no Hugging Face model
        # card at all (they aren't hosted there), so hf_model is {}. The old
        # code returned the neutral 0.5 default for every dimension in that
        # case, which is wrong: a proprietary API-only model from a known
        # Big Tech org is essentially never open on data, weights, base
        # model, or deployment. We use the org_inherit-derived author (via
        # self.organisation.name, which OrganisationSovereigntyScore.detect_*
        # sets from the model_id's namespace even when hf_model is empty) to
        # catch this case and assign the same "closed proprietary" profile
        # the GROUND_TRUTH table uses for openai/gpt-4 etc., rather than
        # silently falling back to neutral scores.
        org_slug = (self.organisation.name or "").lower()
        org_is_big_tech = any(bt in org_slug for bt in BIG_TECH_ORGS)

        if not hf_model:
            if org_is_big_tech:
                # Mirror the GROUND_TRUTH "closed API-only" profile used for
                # known closed models from this org type. High confidence
                # because "Big Tech org + zero public model card" is itself
                # strong, unambiguous evidence of an API-only product.
                scores = {
                    "Training Data Independence":  0.05,
                    "Compute Independence":        0.05,
                    "Weight Ownership & Access":    0.05,
                    "Base Model Dependency":        0.50,  # genuinely unknown — could be scratch or fine-tune
                    "Deployment Independence":      0.02,
                    "Organisational Independence":  0.10,
                    "Jurisdictional Risk":          JURISDICTION_SCORE.get(self.organisation.country, 0.15),
                }
                confidence = {
                    "Training Data Independence":  0.55,
                    "Compute Independence":        0.55,
                    "Weight Ownership & Access":    0.60,
                    "Base Model Dependency":        0.20,  # left genuinely uncertain
                    "Deployment Independence":      0.70,
                    "Organisational Independence":  0.70,
                    "Jurisdictional Risk":          0.55 if self.organisation.country != "–" else 0.30,
                }
            return scores, confidence

        author        = (hf_model.get("author") or "").lower()
        model_id_lower= (hf_model.get("id") or self.model_id).lower()
        tags          = [t.lower() for t in (hf_model.get("tags") or [])]
        tag_text      = " ".join(tags)

        card_data = hf_model.get("cardData")
        if not isinstance(card_data, dict):
            card_data = {}

        readme = ""
        raw_readme = hf_model.get("readme")
        if isinstance(raw_readme, str):
            readme = raw_readme.lower()

        # ── License ──────────────────────────────────────────────────────────
        raw_license = hf_model.get("license") or ""
        if isinstance(raw_license, list):
            licenses = [str(x).lower() for x in raw_license]
        else:
            licenses = [str(raw_license).lower()] if raw_license else []

        has_open_license       = any(any(ol in lic for ol in OPEN_LICENSES)       for lic in licenses)
        has_restricted_license = any(any(rl in lic for rl in RESTRICTED_OPEN_LICENSES) for lic in licenses)
        has_no_license         = not licenses or licenses == [""]

        # ── Base model ───────────────────────────────────────────────────────
        base_model = card_data.get("base_model") or hf_model.get("base_model") or ""
        if isinstance(base_model, list):
            base_model = base_model[0] if base_model else ""
        base_model = str(base_model).lower().strip()

        fine_tune_tags = {"finetuned", "fine-tuned", "fine_tuned", "derived",
                          "instruction-tuned", "rlhf", "lora", "qlora"}
        has_finetune_tag = any(ft in t for t in tags for ft in fine_tune_tags)

        # ── Strategy 2: sibling / Spaces / param-count signals ───────────────
        has_local_siblings  = _has_local_run_siblings(hf_model)
        has_spaces_link     = _has_spaces(hf_model)
        param_count         = _extract_param_count(hf_model)
        datasets_score      = _score_datasets_field(card_data)
        readme_compute      = _score_readme_compute(readme)
        readme_data         = _score_readme_data(readme)

        # A model with no declared base_model and params < ~100B is very likely a
        # fine-tune that omitted the field, not a genuine scratch-trained model.
        # True scratch-trained models are either enormous (>70B) or explicitly declare
        # "trained from scratch" in their card.  We treat anything under 100B with no
        # declared base as a probable fine-tune unless the README says otherwise.
        scratch_hint = any(
            phrase in readme
            for phrase in ["trained from scratch", "pretraining from scratch",
                           "pre-trained from scratch", "no base model"]
        )
        is_probably_finetune = (
            has_finetune_tag
            or (
                not base_model
                and not scratch_hint
                and param_count is not None
                and param_count < 100_000_000_000   # under 100B → assume fine-tune
            )
        )

        # ── 1. Training Data Independence ─────────────────────────────────────
        if datasets_score is not None:
            scores["Training Data Independence"]     = datasets_score
            confidence["Training Data Independence"] = 0.65
        elif readme_data is not None:
            scores["Training Data Independence"]     = readme_data[0]
            confidence["Training Data Independence"] = readme_data[1]
        elif "fully open" in readme or "open data" in tag_text:
            scores["Training Data Independence"]     = 0.85
            confidence["Training Data Independence"] = 0.60
        elif "transparent" in tag_text or "open" in tag_text:
            scores["Training Data Independence"]     = 0.65
            confidence["Training Data Independence"] = 0.50
        elif has_open_license:
            scores["Training Data Independence"]     = 0.55
            confidence["Training Data Independence"] = 0.40
        else:
            scores["Training Data Independence"]     = 0.40
            confidence["Training Data Independence"] = 0.35

        # ── 2. Compute Independence ───────────────────────────────────────────
        if readme_compute is not None:
            scores["Compute Independence"]     = readme_compute[0]
            confidence["Compute Independence"] = readme_compute[1]
        elif any(k in model_id_lower or k in author
                 for k in ["swiss-ai", "ai-sweden", "cscs", "epfl"]):
            scores["Compute Independence"]     = 0.80
            confidence["Compute Independence"] = 0.65
        elif any(k in author for k in BIG_TECH_ORGS):
            scores["Compute Independence"]     = 0.10
            confidence["Compute Independence"] = 0.70
        else:
            scores["Compute Independence"]     = 0.45
            confidence["Compute Independence"] = 0.25   # genuinely uncertain

        # ── 3. Weight Ownership & Access ──────────────────────────────────────
        if has_open_license and not has_restricted_license:
            scores["Weight Ownership & Access"]     = 0.85
            confidence["Weight Ownership & Access"] = 0.75
        elif has_restricted_license:
            scores["Weight Ownership & Access"]     = 0.50
            confidence["Weight Ownership & Access"] = 0.70
        elif has_no_license:
            scores["Weight Ownership & Access"]     = 0.25
            confidence["Weight Ownership & Access"] = 0.40
        else:
            scores["Weight Ownership & Access"]     = 0.15
            confidence["Weight Ownership & Access"] = 0.65

        # ── 4. Base Model Dependency ──────────────────────────────────────────
        if base_model:
            # Check if the base model itself is in the ground-truth table.
            gt_base, _ = _gt_family_lookup(base_model)
            if gt_base is not None:
                # Use the base model's own Weight Ownership score as a proxy.
                bm_dep = gt_base.get("Weight Ownership & Access", 0.5)
                scores["Base Model Dependency"]     = round(bm_dep * 0.9, 3)
                confidence["Base Model Dependency"] = 0.80
            elif any(bt in base_model for bt in ["openai", "gpt", "claude", "gemini", "palm"]):
                scores["Base Model Dependency"]     = 0.10
                confidence["Base Model Dependency"] = 0.80
            elif any(bt in base_model for bt in ["llama", "mistral", "falcon", "qwen"]):
                scores["Base Model Dependency"]     = 0.35
                confidence["Base Model Dependency"] = 0.75
            else:
                scores["Base Model Dependency"]     = 0.40
                confidence["Base Model Dependency"] = 0.60
        elif is_probably_finetune:
            # Small model, no declared base — assume fine-tune of something open
            scores["Base Model Dependency"]     = 0.35
            confidence["Base Model Dependency"] = 0.45
        else:
            # Likely trained from scratch (large model, no base declared)
            scores["Base Model Dependency"]     = 0.80
            confidence["Base Model Dependency"] = 0.50

        # ── 5. Deployment Independence ────────────────────────────────────────
        if has_local_siblings:
            # GGUF/ONNX files present → can definitely run locally
            scores["Deployment Independence"]     = 0.95
            confidence["Deployment Independence"] = 0.90
        elif has_open_license and not has_restricted_license:
            base_score = 0.90
            if has_spaces_link:
                base_score = min(1.0, base_score + 0.05)
            scores["Deployment Independence"]     = base_score
            confidence["Deployment Independence"] = 0.75
        elif has_restricted_license:
            scores["Deployment Independence"]     = 0.60
            confidence["Deployment Independence"] = 0.65
        elif has_no_license:
            scores["Deployment Independence"]     = 0.20
            confidence["Deployment Independence"] = 0.40
        else:
            scores["Deployment Independence"]     = 0.10
            confidence["Deployment Independence"] = 0.60

        # ── 6. Organisational Independence ────────────────────────────────────
        if any(bt in author for bt in BIG_TECH_ORGS):
            scores["Organisational Independence"]     = 0.05
            confidence["Organisational Independence"] = 0.85
        elif any(pi in author for pi in PUBLIC_INSTITUTION_HINTS):
            scores["Organisational Independence"]     = 0.90
            confidence["Organisational Independence"] = 0.75
        elif "university" in author or "institute" in author or "research" in author:
            scores["Organisational Independence"]     = 0.80
            confidence["Organisational Independence"] = 0.60
        else:
            scores["Organisational Independence"]     = 0.55
            confidence["Organisational Independence"] = 0.40

        # ── 7. Jurisdictional Risk ────────────────────────────────────────────
        country = self.organisation.country
        scores["Jurisdictional Risk"]     = JURISDICTION_SCORE.get(country, 0.40)
        confidence["Jurisdictional Risk"] = 0.65 if country != "–" else 0.25

        return scores, confidence

    # ------------------------------------------------------------------
    # Strategy 3: dimension-specific web fetch
    # ------------------------------------------------------------------

    def _fetch_dimension_docs(
        self,
        category: str,
        model_name: str,
        org_name: str,
        user_agent: str = "Sovereignty-Pipeline/1.0",
        top_k: int = 3,
    ) -> list[dict]:
        """
        Fetch web documents specifically relevant to *category* using the
        targeted query templates in DIMENSION_QUERIES.

        Falls back gracefully to an empty list if pipeline or network
        is unavailable.
        """
        if not _PIPELINE_AVAILABLE:
            return []

        templates = DIMENSION_QUERIES.get(category, [])
        all_docs: list[dict] = []
        seen_urls: set[str] = set()

        for template in templates[:2]:   # at most 2 queries per dimension
            query = (
                template
                .replace("{model}", model_name)
                .replace("{org}", org_name or model_name)
            )
            try:
                docs = fetch_web_evidence(
                    query,
                    user_agent=user_agent,
                    top_k_per_query=top_k,
                    delay_between_requests=0.5,
                    verbose=self.verbose,
                )
                for d in docs:
                    url = d.get("url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_docs.append(d)
            except Exception as exc:
                self._log_error("_fetch_dimension_docs", exc, context={"query": query})

        return all_docs

    def _score_dimension_with_llm(
        self,
        category: str,
        docs: list[dict],
        model_name: str,
        num_attempts: int = 2,
    ) -> Optional[dict]:
        """
        Run a single-dimension LLM scoring call against pre-filtered documents.
        Returns a dict with keys: score, confidence, evidence.
        """
        method = "_score_dimension_with_llm"
        if not _PIPELINE_AVAILABLE or not os.getenv("PUBLICAI_KEY"):
            return None

        filtered = [d for d in docs if _is_relevant(d.get("extracted", ""), category)] or docs
        if not filtered:
            return None

        model_lower   = model_name.lower()
        source_blocks: list[str] = []
        for j, d in enumerate(filtered[:4]):
            raw = d.get("extracted") or ""
            sentences = re.split(r"(?<=[.!?])\s+", raw)
            relevant  = [
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
            f"Definition:\n{_CATEGORY_DEFINITIONS.get(category, '')}\n\n"
            f"Instructions:\n"
            f"- Find a sentence directly relevant to the question.\n"
            f"- Copy it EXACTLY as it appears.\n"
            f"- Score 1.0 = fully sovereign on this dimension, 0.0 = fully dependent on Big Tech.\n\n"
            f"Return this exact JSON:\n"
            f'{{\n'
            f'  "score": <float 0.0-1.0>,\n'
            f'  "confidence": <float 0.0-1.0>,\n'
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

            score      = _parse_float(parsed.get("score"))      or 0.5
            conf       = _parse_float(parsed.get("confidence")) or 0.0
            quote_text = (parsed.get("quote") or "").strip()
            rationale  = (parsed.get("rationale") or "").strip()

            evidence: list[dict] = []
            if quote_text and len(quote_text) > 20:
                verified_url = _verify_quote(quote_text, filtered)
                if verified_url is not None:
                    evidence.append({
                        "quote":     quote_text,
                        "url":       verified_url or parsed.get("url", ""),
                        "rationale": rationale,
                        "verified":  True,
                    })

            if not evidence:
                best = self._pick_best_sentence(filtered, category, model_name)
                if best:
                    best["verified"] = False
                    evidence.append(best)

            return {"score": score, "confidence": conf, "evidence": evidence}

        except Exception as exc:
            self._log_error(method, exc, context={"category": category})
            best = self._pick_best_sentence(filtered, category, model_name)
            return {
                "score":      0.5,
                "confidence": 0.0,
                "evidence":   [{**best, "verified": False}] if best else [],
            }

    def _fetch_jurisdiction_score(
        self, model_id: str, org_name: str
    ) -> Optional[float]:
        """
        Dedicated jurisdiction web query fired when country detection returns '–'.
        Returns a JURISDICTION_SCORE float if a country can be identified,
        else None.
        """
        if not _PIPELINE_AVAILABLE:
            return None
        query = f'"{org_name or model_id}" headquarters incorporated country domicile'
        try:
            docs = fetch_web_evidence(
                query,
                user_agent="Sovereignty-Pipeline/1.0",
                top_k_per_query=3,
                delay_between_requests=0.5,
                verbose=self.verbose,
            )
        except Exception:
            return None

        blob = " ".join(d.get("extracted", "") for d in docs).lower()
        for country_name, score in sorted(
            JURISDICTION_SCORE.items(), key=lambda x: x[1], reverse=True
        ):
            if country_name.lower() in blob:
                self._log(
                    f"Jurisdiction inferred from web: {country_name}", method="_fetch_jurisdiction_score"
                )
                # Update the parent org country if it's still unknown
                if self.organisation.country == "–":
                    self.organisation.country = country_name
                return score
        return None

    def _pick_best_sentence(
        self, docs: list[dict], category: str, model_name: str
    ) -> Optional[dict]:
        model_lower  = model_name.lower()
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
                        return {
                            "quote":     s,
                            "url":       d.get("url", ""),
                            "rationale": f"Best available text for: {category}",
                        }
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
                "category":   c,
                "score":      round(float(self.category_scores.get(c, 0.5)), 3),
                "confidence": round(float(self.category_confidence.get(c, 0.3)), 3),
                "weight":     round(self.weights.get(c, 1.0 / len(CATEGORIES)), 3),
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
        gt_note = (
            f"Ground-truth match type: {self.gt_match_type}. "
            "Some dimensions were inherited from known entries."
            if self.used_ground_truth or self.gt_match_type
            else "Scores derived from HF metadata and web evidence."
        )

        # Build a sorted view so the prompt can reference most/least influential dims easily
        dims_sorted = sorted(dims, key=lambda d: d["weight"] * abs(d["score"] - 0.5), reverse=True)

        BAD_EXAMPLE = (
            "Compute Independence scored high because the model was trained on proprietary "
            "infrastructure owned by the developing organization, scoring 0.89. Licensing "
            "Independence also scored high (0.95) as it is released under a permissive licence "
            "with no Big Tech involvement."
        )
        GOOD_EXAMPLE = (
            "The model's weights are gated behind a request-access form on a platform controlled "
            "by the originating lab, meaning any downstream operator can have access revoked "
            "unilaterally — a hard constraint on redistribution rights regardless of what the "
            "licence text says. Training ran entirely on leased hyperscaler capacity, so the "
            "compute dependency is structural rather than incidental; a commercial dispute or "
            "policy change at the cloud provider would halt any further development."
        )

        prompt = f"""
            You are a technical analyst. Your job is to explain the *causal reasoning* behind an AI sovereignty
            assessment — specifically, what real-world facts, licence terms, and architectural decisions drove
            the result for {self.model_id}.

            Definition of AI sovereignty:
            A sovereign AI model is one that can be accessed, deployed, modified, and operated without reliance
            on, control by, or enforced constraints from dominant external providers (e.g. Big Tech platforms).
            This includes independence across infrastructure, distribution channels, licensing, weights access,
            and operational control.

            {gt_note}

            Category scores (0=not sovereign, 1=fully sovereign), sorted by influence on the final score:
            {json.dumps(dims_sorted, indent=2)}

            Web evidence:
            {evidence_section}

            ---
            HARD RULES — violating any of these makes the output wrong:
            1. Do not mention any score, number, or percentage. The reader has the table.
            2. Do not name a category and then describe what that category measures.
            3. Do not use bullet points, numbered lists, headers, or markdown.
            4. Every factual claim must trace back to the evidence above or universally known facts about this model.
            5. Pick the 2-3 categories with most leverage on the final score and explain only those.
            6. If signals conflict (e.g. open weights but cloud-only deployment), say so and explain which dominates and why.

            Here is the difference between a BAD answer and a GOOD answer for a fictional model:

            BAD (restates scores, names categories abstractly, no causal chain):
            {BAD_EXAMPLE}

            GOOD (concrete facts, causal chain, no numbers, plain prose):
            {GOOD_EXAMPLE}

            Now write the GOOD-style explanation for {self.model_id}. 5-7 sentences. No preamble.
        """
        try:
            return ask_publicai(prompt=prompt, user_agent=user_agent)
        except Exception as exc:
            self._log_error("_explain_score", exc)
            return self._fallback_explanation()

    def _fallback_explanation(self) -> str:
        parts = [f"{self.model_id} sovereignty score: {self.overall_score:.2f}/100."]
        if self.gt_match_type:
            parts.append(f"Ground-truth match type: {self.gt_match_type}.")
        for c in CATEGORIES:
            v    = self.category_scores.get(c, 0.5)
            conf = self.category_confidence.get(c, 0.3)
            flag = " [low confidence]" if conf < 0.4 else ""
            parts.append(f"{c}: {v:.3f}{flag}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Category definitions (used in LLM prompts)
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
        "weights are proprietary and access can be revoked at any time (API-only)."
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
# Country / org-type detection (unchanged from original, kept intact)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _build_keyword_variants(kw: str) -> list[str]:
    raw        = kw.lower().strip()
    normalised = _normalise(kw)
    no_space   = normalised.replace(" ", "")
    hyphenated = normalised.replace(" ", "-")
    underscored= normalised.replace(" ", "_")
    return list({raw, normalised, no_space, hyphenated, underscored})


_KW_VARIANTS: dict[str, list[str]] = {
    kw: _build_keyword_variants(kw) for kw in COUNTRY_KEYWORDS
}

_ORG_SLUG_SUFFIXES: frozenset[str] = frozenset({
    "ai", "hq", "labs", "lab", "inc", "ltd", "llc", "corp", "co",
    "group", "team", "models", "model", "research", "institute",
})


def _match_country_word_boundary(text: str) -> Optional[str]:
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
    norm = _normalise(text)
    if not norm:
        return None
    for token in sorted(set(norm.split()), key=len, reverse=True):
        hit = _match_country_word_boundary(token)
        if hit:
            return hit
    return None


def _match_country_slug_segment(segment: str) -> Optional[str]:
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
    if not slug:
        return None
    slug_lower = slug.lower()
    segments   = [s for s in re.split(r"[-_./]+", slug_lower) if s]
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
    for kw in sorted(COUNTRY_KEYWORDS, key=len, reverse=True):
        for variant in _KW_VARIANTS[kw]:
            if len(variant) < 4:
                continue
            for form in {variant, variant.replace(" ", "-"), variant.replace(" ", "_")}:
                if slug_lower == form:
                    return COUNTRY_KEYWORDS[kw]
                if re.search(rf"(^|[-_.]){re.escape(form)}($|[-_.])", slug_lower):
                    return COUNTRY_KEYWORDS[kw]
    return None


def _match_country(text: str) -> Optional[str]:
    if not text:
        return None
    for matcher in (_match_country_word_boundary, _match_country_tokens, _match_country_from_slug):
        hit = matcher(text)
        if hit:
            return hit
    return None


def _collect_hf_country_text(hf_model: dict, org: str) -> str:
    parts: list[str] = [org, hf_model.get("author") or "", hf_model.get("id") or ""]
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
# OrganisationSovereigntyScore
# ---------------------------------------------------------------------------

@dataclass
class OrganisationSovereigntyScore:
    """
    Aggregates sovereignty scores for all models of one organisation.
    """

    name:              str
    organisation_type: str  = "Independent"
    country:           str  = "–"
    metadata:          dict = field(default_factory=dict)

    _models: list[ModelSovereigntyScore] = field(
        default_factory=list, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def add_model(self, model: ModelSovereigntyScore) -> None:
        if model.organisation is not self:
            raise ValueError(f"Model '{model.model_id}' is linked to a different organisation.")
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
        effective_id = model_id or (hf_model.get("id") if hf_model else "") or ""
        namespace    = effective_id.split("/")[0] if "/" in effective_id else effective_id

        if not hf_model:
            hit = _match_country_from_slug(namespace) if namespace else None
            return hit if hit else "–"

        author = (hf_model.get("author") or "").strip()
        org    = effective_id.split("/")[0] if "/" in effective_id else author

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
            elif any(k in blob for k in ["non-profit", "nonprofit", "charity", "foundation", "ngo"]):
                org_type = "Non-profit"
            elif any(k in blob for k in ["community", "collective", "open-source"]):
                org_type = "Community"
        except Exception:
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
                "name":                    self.name,
                "organisation_type":       self.organisation_type,
                "country":                 self.country,
                "metadata":                self.metadata,
                "aggregate":               self.score_summary(),
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
            m.gt_match_type       = m_data.get("gt_match_type")
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
        name=              author,
        organisation_type= OrganisationSovereigntyScore.detect_org_type(hf_model or {}),
        country=           OrganisationSovereigntyScore.detect_country(hf_model or {}, model_id=model_id),
        metadata={
            "source":       "public-ai sovereignty pipeline",
            "version":      "0.3.0",
            "uses_web":     use_web,
            "uses_llm_web": use_llm_web,
        },
    )

    model_score = org.create_model(model_id)
    model_score.evaluate(use_web=use_web, use_llm_web=use_llm_web, verbose=verbose)
    return org, model_score