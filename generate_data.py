import json
import os

from pipeline.sovereignty_score import evaluate_model_for_hf, explain_sovereignty_score

USE_WEB = True
USE_LLM_WEB = True

MODELS = [
    # GPT-SW3 (multiple sizes & variants; model cards exist) :contentReference[oaicite:0]{index=0}
    "AI-Sweden-Models/gpt-sw3-20b-instruct",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "aisingapore/Apertus-SEA-LION-v4-8B-IT-GGUF",
    "BSC-LT/salamandra-7b-instruct",
    "sdaia/allam-3b-instruct",
    "sarvamai/sarvam-2b",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Molmo-7B-O-0924",
    "EleutherAI/pythia-12b"
]

output_path = os.path.join("data", "models.json")
os.makedirs("data", exist_ok=True)

data = []

for model_id in MODELS:
    print(f"Scoring {model_id} (use_web={USE_WEB})...")

    try:
        score = evaluate_model_for_hf(
            model_id,
            use_web=USE_WEB,
            use_llm_web=USE_LLM_WEB,
        )
        explain = explain_sovereignty_score(score)

        entry = {
            "model": model_id.split("/")[-1],
            "score": score,
            "explain": explain,
        }

        data.append(entry)

        # ✅ Write after each iteration (atomic-ish)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, output_path)

    except Exception as e:
        print(f"Error processing {model_id}: {e}")
        continue

print(f"Wrote {len(data)} model(s) to {output_path}")