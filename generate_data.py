
import json
import os
 
from pipeline.sovereignty_score import OrganisationSovereigntyScore, evaluate_model
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
USE_WEB = True
USE_LLM_WEB = True
 
MODELS = [
    # "AI-Sweden-Models/gpt-sw3-20b-instruct",
    # "swiss-ai/Apertus-8B-Instruct-2509",
    "openai/gpt-4o",
    # "anthropic/claude-3-opus",
    # "aisingapore/Apertus-SEA-LION-v4-8B-IT-GGUF",
    # "BSC-LT/salamandra-7b-instruct",
    # "sdaia/allam-3b-instruct",
    # "sarvamai/sarvam-2b",
    # "allenai/Olmo-3-7B-Instruct",
    # "allenai/Molmo-7B-O-0924",
    # "EleutherAI/pythia-12b",
]

# MODELS = [
#     # =========================
#     # 🌐 Frontier / Closed Models (APIs)
#     # =========================
#     "openai/gpt-4o",
#     "openai/gpt-4.1",
#     "openai/gpt-5",
#     "anthropic/claude-3-opus",
#     "anthropic/claude-3.5-sonnet",
#     "google/gemini-1.5-pro",
#     "google/gemini-1.5-flash",
#     "google/gemini-2.0",
#     "xai/grok-1.5",
#     "mistralai/mistral-large",
#     "amazon/nova-pro",

#     # =========================
#     # 🧠 Open-Weight / Open Models
#     # =========================
#     "meta-llama/llama-3-70b-instruct",
#     "meta-llama/llama-3-8b-instruct",
#     "meta-llama/llama-2-70b",
#     "mistralai/mistral-7b-instruct",
#     "mistralai/mixtral-8x7b-instruct",
#     "mistralai/mixtral-8x22b",
#     "google/gemma-7b",
#     "google/gemma-2-9b",
#     "google/gemma-2-27b",
#     "tiiuae/falcon-40b",
#     "tiiuae/falcon-180b",
#     "databricks/dbrx-instruct",
#     "togethercomputer/redpajama-incite-7b",
#     "bigscience/bloom",
#     "bigscience/bloomz",

#     # =========================
#     # 🇪🇺 European / Sovereign AI
#     # =========================
#     "mistralai/mistral-small",
#     "mistralai/mistral-medium",
#     "AlephAlpha/luminous-supreme",
#     "LAION/leoLM-13b",
#     "OpenGPT-X/Teuken-7B-instruct",
#     "BSC-LT/salamandra-7b-instruct",
#     "AI-Sweden-Models/gpt-sw3-20b-instruct",
#     "swiss-ai/Apertus-8B-Instruct-2509",

#     # =========================
#     # 🌏 Asia (Sovereign + Regional)
#     # =========================
#     "qwen/Qwen2-72B-Instruct",
#     "qwen/Qwen2-7B-Instruct",
#     "baichuan-inc/Baichuan2-13B-Chat",
#     "01-ai/Yi-34B-Chat",
#     "deepseek-ai/deepseek-llm-67b-chat",
#     "deepseek-ai/deepseek-coder-33b-instruct",
#     "aisingapore/Apertus-SEA-LION-v4-8B-IT",
#     "sarvamai/sarvam-2b",
#     "sdaia/allam-3b-instruct",

#     # =========================
#     # 🇺🇸 Research / Open Science
#     # =========================
#     "allenai/OLMo-7B-Instruct",
#     "allenai/Molmo-7B-O",
#     "EleutherAI/gpt-neox-20b",
#     "EleutherAI/pythia-12b",
#     "stanford-crfm/helium-1-preview",

#     # =========================
#     # 🧑‍💻 Coding-Specialised Models
#     # =========================
#     "bigcode/starcoder2-15b",
#     "codellama/CodeLlama-70b-Instruct",
#     "deepseek-ai/deepseek-coder-6.7b",
#     "wizardlm/WizardCoder-15B",

#     # =========================
#     # 🖼️ Multimodal Models
#     # =========================
#     "openai/gpt-4o-mini",
#     "google/gemini-pro-vision",
#     "llava-hf/llava-1.5-13b",
#     "adept/fuyu-8b",

#     # =========================
#     # 🪶 Smaller / Edge Models
#     # =========================
#     "microsoft/phi-3-mini",
#     "microsoft/phi-3-medium",
#     "google/gemma-2b",
#     "TinyLlama/TinyLlama-1.1B-Chat",

#     # =========================
#     # 🔬 Experimental / Emerging
#     # =========================
#     "inflection/pi",
#     "perplexity/pplx-70b-online",
#     "cohere/command-r",
#     "cohere/command-r-plus",
# ]
 
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