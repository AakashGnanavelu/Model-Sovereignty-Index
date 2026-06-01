
import json
import os
 
from pipeline.sovereignty_score import OrganisationSovereigntyScore, evaluate_model
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
USE_WEB = True
USE_LLM_WEB = True

MODELS = [
    # =========================
    # 🌐 Big Tech / Frontier (Baseline)
    # =========================
    "openai/gpt-5",
    "anthropic/claude-3-opus",
    "google/gemini-2.0",
    "xai/grok-1.5",

    # =========================
    # 🇪🇺 European Sovereign AI
    # =========================
    "mistralai/mistral-large",
    "mistralai/mixtral-8x22b",
    "AlephAlpha/luminous-supreme",
    "OpenGPT-X/Teuken-7B-instruct",
    "LAION/leoLM-13b",
    "AI-Sweden-Models/gpt-sw3-20b-instruct",
    "swiss-ai/Apertus-8B-Instruct-2509",

    # =========================
    # 🌏 Asia Sovereign / Regional
    # =========================
    "qwen/Qwen2-72B-Instruct",
    "deepseek-ai/deepseek-llm-67b-chat",
    "01-ai/Yi-34B-Chat",
    "baichuan-inc/Baichuan2-13B-Chat",
    "aisingapore/Apertus-SEA-LION-v4-8B-IT",
    "sarvamai/sarvam-2b",

    # =========================
    # 🌍 Open / Research (Non-Big Tech Control)
    # =========================
    "databricks/dbrx-instruct",
    "tiiuae/falcon-180b",
    "allenai/OLMo-7B-Instruct",
]
 
OUTPUT_PATH = os.path.join("data", "models.json")
os.makedirs("data", exist_ok=True)
 
# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------
 
# Accumulate all OrganisationSovereigntyScore objects so we can write a
# single structured JSON containing both org metadata and model scores.
organisations: list[OrganisationSovereigntyScore] = []
 
for model_id in MODELS:
    print(f"Scoring {model_id} (use_web={USE_WEB}, use_llm_web={USE_LLM_WEB})...")
    try:
        # evaluate_model returns (org, model_score) — both already linked
        org, model_score = evaluate_model(
            model_id,
            use_web=USE_WEB,
            use_llm_web=USE_LLM_WEB,
            verbose=True
        )
 
        # Generate a natural-language explanation and store it on the model
        model_score.generate_explanation()
 
        organisations.append(org)
 
        # Write after every successful model so progress survives a crash
        payload = [o.to_dict() for o in organisations]
        tmp_path = OUTPUT_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, OUTPUT_PATH)
 
        print(
            f"  ✓ {model_id} → {model_score.overall_score}/100  "
            f"(org: {org.name}, country: {org.country})"
        )
 
    except Exception as e:
        print(f"  ✗ Error processing {model_id}: {e}")
        continue
 
print(f"\nWrote {len(organisations)} model(s) to {OUTPUT_PATH}")