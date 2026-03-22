#!/usr/bin/env python3
"""
CLI: compute model sovereignty score from Hugging Face + optional web.
Usage: python sovereignty_score.py "Apertus" [--web] [--llm] [--json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional: map display names to Hugging Face model IDs for HF fetch
KNOWN_MODELS = {
    "apertus": "swiss-ai/Apertus-70B-2509",
    "gpt-sw3": "AI-Sweden-Models/gpt-sw3-40b",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute model sovereignty score (0-100).")
    parser.add_argument("model", help='Model name (e.g. "Apertus") or Hugging Face ID (e.g. swiss-ai/Apertus-70B-2509)')
    parser.add_argument("--web", action="store_true", help="Fetch web evidence (search + scrape)")
    parser.add_argument("--llm", action="store_true", help="Use LLM to score from web docs (needs PUBLICAI_KEY)")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    args = parser.parse_args()

    model_input = args.model.strip()
    if "/" in model_input:
        hf_model_id = model_input
        model_name = model_input.split("/")[-1].replace("-", " ")
    else:
        model_name = model_input
        hf_model_id = KNOWN_MODELS.get(model_name.lower())

    # Fetch Hugging Face
    hf_model = None
    if hf_model_id:
        _log("Fetching Hugging Face model metadata...")
        from pipeline.sources.huggingface import fetch_huggingface_model
        hf_model = fetch_huggingface_model(hf_model_id)
        if hf_model:
            _log(f"  Found: {hf_model.get('id', hf_model_id)}")
        else:
            _log("  No HF data (model not found or error).")
    else:
        _log("No Hugging Face ID for this model; using web-only for scoring.")

    # Optional web evidence
    web_docs = []
    if args.web:
        _log("Fetching web evidence (this may take a minute)...")
        from pipeline.sources.web import fetch_web_evidence
        web_docs = fetch_web_evidence(model_name, top_k_per_query=3, delay_between_requests=1.0)
        _log(f"  Collected {len(web_docs)} document(s).")

    _log("Computing sovereignty score...")
    from pipeline.sovereignty_score import compute_sovereignty_score, CATEGORIES, explain_sovereignty_score
    score_100, category_scores = compute_sovereignty_score(
        hf_model,
        web_docs=web_docs if web_docs else None,
        model_name=model_name,
        use_llm_web=args.llm,
    )

    if args.json:
        out = {
            "model": model_name,
            "huggingface_id": hf_model_id,
            "sovereignty_score": score_100,
            "category_scores": category_scores,
            "sources": {"huggingface": hf_model is not None, "web_docs_count": len(web_docs),
            "web_docs": web_docs},
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Sovereignty score: {score_100}/100")
        eval_like = {
            "model_id": hf_model_id or model_name,
            "value": score_100,
            "categories": category_scores,
            "metadata": {"uses_web": bool(web_docs), "uses_llm_web": bool(args.llm)},
        }
        print(explain_sovereignty_score(eval_like))
        print("Category breakdown:")
        for c in CATEGORIES:
            print(f"  {c}: {category_scores.get(c, 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
