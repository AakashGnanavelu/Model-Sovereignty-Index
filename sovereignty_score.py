#!/usr/bin/env python3
"""
CLI: compute model sovereignty score

Usage:
  python sovereignty_score.py "swiss-ai/Apertus-70B-2509"
  python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --web --llm --explain
  python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --json
  python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --json --explain
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute model sovereignty score.")
    parser.add_argument("model", help="Hugging Face model ID")
    parser.add_argument("--web", action="store_true", help="Use web evidence")
    parser.add_argument("--llm", action="store_true", help="Use LLM over web evidence")
    parser.add_argument("--explain", action="store_true", help="Generate explanation")
    parser.add_argument("--json", action="store_true", help="Output full JSON")

    args = parser.parse_args()
    model_id = args.model.strip()

    from pipeline.sovereignty_score import (
        evaluate_model_for_hf,
        explain_sovereignty_score,
    )

    # -------------------------
    # Run evaluation
    # -------------------------
    result = evaluate_model_for_hf(
        model_id,
        use_web=args.web,
        use_llm_web=args.llm,
    )

    score = result.get("value")

    # -------------------------
    # JSON mode (with optional explanation)
    # -------------------------
    if args.json:
        output = dict(result)  # shallow copy

        if args.explain:
            try:
                output["explanation"] = explain_sovereignty_score(result)
            except Exception as e:
                output["explanation"] = f"Failed to generate explanation: {e}"

        print(json.dumps(output, indent=2))
        return 0

    # -------------------------
    # Text output
    # -------------------------
    print(f"{model_id}: {score}/100")

    if args.explain:
        print()
        explanation = explain_sovereignty_score(result)
        print(explanation)

    return 0


if __name__ == "__main__":
    sys.exit(main())