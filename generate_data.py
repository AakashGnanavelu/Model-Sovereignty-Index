
import json
import os
 
from pipeline.sovereignty_score import OrganisationSovereigntyScore, evaluate_model
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
USE_WEB = True
USE_LLM_WEB = True

MODELS = [
    # =========================================================================
    # 🌐 Closed Frontier Models (Big Tech, API-only)
    # =========================================================================
    "openai/gpt-5",
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-opus-4",
    "google/gemini-2-5-pro",
    "xai/grok-3",

    # =========================================================================
    # 🇺🇸 Open-Weight Frontier Models (US Big Tech)
    # =========================================================================
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-3-27b-it",
    "microsoft/phi-4",

    # =========================================================================
    # 🇨🇳 Chinese Open-Weight Models
    # =========================================================================
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen2-72B-Instruct",

    # =========================================================================
    # 🇪🇺 European Sovereign Models
    # =========================================================================
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "aleph-alpha/luminous-supreme",

    # =========================================================================
    # 🇸🇪 Sweden
    # =========================================================================
    "ai-sweden/gpt-sw3-126m",
    "AI-Sweden-Models/gpt-sw3-20b-instruct",

    # =========================================================================
    # 🇨🇭 Switzerland
    # =========================================================================
    "swiss-ai/swissbert",
    "swiss-ai/Apertus-70B-Instruct-2509",

    # =========================================================================
    # 🇬🇧 United Kingdom
    # =========================================================================
    "UKEB/UK-Llama-3.1-8B-Instruct",

    # =========================================================================
    # 🇩🇪 Germany
    # =========================================================================
    "OpenGPT-X/Teuken-7B-instruct-commercial-v0.4",

    # =========================================================================
    # 🇪🇸 Spain
    # =========================================================================
    "BSC-LT/salamandra-7b-instruct",

    # =========================================================================
    # 🇸🇬 Southeast Asia / Singapore
    # =========================================================================
    "aisingapore/sea-lion-v1-7b",
    "aisingapore/sea-lion-v2-8b",
    "aisingapore/sea-lion-v3-8b-instruct",
    "aisingapore/sea-lion-v4-8b-instruct",

    # =========================================================================
    # 🇸🇦 Saudi Arabia
    # =========================================================================
    "ALLaM-AI/ALLaM-3B-Instruct",

    # =========================================================================
    # 🇮🇳 India
    # =========================================================================
    "sarvamai/sarvam-2b-v0.5",

    # =========================================================================
    # 🇦🇪 United Arab Emirates
    # =========================================================================
    "tiiuae/falcon-40b",
    "tiiuae/falcon-mamba-7b",

    # =========================================================================
    # 🏛️ Open Research / Non-Profit Models (US)
    # =========================================================================
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-neox-20b",
    "allenai/OLMo-1B",
    "allenai/OLMo-7B-0724-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/Molmo-7B-D",
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