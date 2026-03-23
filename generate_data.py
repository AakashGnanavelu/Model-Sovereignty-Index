import json
import os

from pipeline.sovereignty_score import evaluate_model_for_hf, explain_sovereignty_score

USE_WEB = True
USE_LLM_WEB = True

MODELS = [
    # GPT-SW3 (multiple sizes & variants; model cards exist) :contentReference[oaicite:0]{index=0}
    "AI-Sweden-Models/gpt-sw3-126m",
    "AI-Sweden-Models/gpt-sw3-126m-instruct",
    "AI-Sweden-Models/gpt-sw3-356m",
    "AI-Sweden-Models/gpt-sw3-1.3b",
    "AI-Sweden-Models/gpt-sw3-1.3b-instruct",
    "AI-Sweden-Models/gpt-sw3-20b-instruct",

    "swiss-ai/Apertus-8B-Instruct-2509",

    # # Additional Apertus variants with cards
    "unsloth/Apertus-70B-Instruct-2509-GGUF",
    "cpatonn/Apertus-8B-Instruct-2509-GPTQ-4bit",
    "RedHatAI/Apertus-70B-Instruct-2509-FP8-dynamic",
    "safouaneelg/Apertus-8B-Instruct-2509-AQUA-RAT-SFT",

    # # SEA-LION variants with model cards (v4 example) :contentReference[oaicite:2]{index=2}
    "aisingapore/Apertus-SEA-LION-v4-8B-IT-GGUF",

    # # Salamandra models (official Hugging Face cards exist)
    "BSC-LT/salamandra-2b",
    "BSC-LT/salamandra-2b-instruct",
    "BSC-LT/salamandra-7b",
    "BSC-LT/salamandra-7b-instruct",

    # Allam 3B (publicly uploaded models)
    "sdaia/allam-3b-base",
    "sdaia/allam-3b-instruct",

    # Sarvam (published model cards on HF)
    "sarvamai/sarvam-1",
    "sarvamai/sarvam-2b",
    "sarvamai/sarvam-m",

    # OLMo 3 (AllenAI public releases)
    "allenai/Olmo-3-1125-32B",
    "allenai/Olmo-3-32B-Think",
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Think",
    "allenai/Olmo-3-7B-Instruct",

    # # MOLMO (AllenAI)
    "allenai/Molmo-7B-O-0924",

    # Pythia (EleutherAI official family)
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
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