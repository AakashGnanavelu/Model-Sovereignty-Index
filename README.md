## Public AI Sovereignty Score

View Dashboard from this [link](https://aakashgnanavelu.github.io/Model-Sovereignty-Index/).

This repo contains a small pipeline that estimates a **model sovereignty score** using:

- **Hugging Face Hub metadata** (license, author/organisation, tags, base model info)
- Optional **web evidence** (search + scraping) and an optional LLM pass over that evidence

The goal is to provide a **simple, reproducible heuristic** you can use as a **custom evaluation metric** on Hugging Face model cards – not a definitive judgement of legal or political sovereignty.

---

### What is the sovereignty score?

For a given model, we compute:

- **Overall sovereignty score**: a value in **\[0, 100\]**
- **Per‑category scores**: values in **\[0, 1\]** for:
  - `Is the training data private?`
  - `Is the model trained locally?`
  - `Does the company have full control of the model weights?`
  - `Is the model trained on a different model?`
  - `Is the model weights private?`
  - `Is there country specific knowledge?`

By default, the score is derived from **Hugging Face Hub metadata only**. If you enable the optional web components, we also look at public web pages about the model and can blend in an LLM‑based judgement.

---

### Installation

From the repo root:

```bash
pip install -r requirements.txt
```

Optional extras (used only if you enable `use_web=True` or `--web`):

- `ddgs` – DuckDuckGo search
- `trafilatura` – HTML extraction
- `PUBLICAI_KEY` env var – if you want LLM‑based scoring over web docs

If you just want the **Hub‑only** score, `requests` is enough.

---

### Programmatic API for Hugging Face evaluations

The main helper for building a **model-level** Hugging Face evaluation result is:

```python
from pipeline.sovereignty_score import evaluate_model_for_hf

result = evaluate_model_for_hf("swiss-ai/Apertus-70B-2509", use_web=False)
print(result)
```

Example output:

```json
{
  "model_id": "swiss-ai/Apertus-70B-2509",
  "metric_name": "sovereignty_score",
  "metric_type": "custom",
  "value": 62.5,
  "categories": {
    "Is the training data private?": 0.5,
    "Is the model trained locally?": 0.7,
    "Does the company have full control of the model weights?": 0.3,
    "Is the model trained on a different model?": 0.8,
    "Is the model weights private?": 0.8,
    "Is there country specific knowledge?": 0.65
  },
  "sources": [
    "https://example.com/evidence-1",
    "https://example.com/evidence-2"
  ],
  "organisation_type": "State-backed",
  "country": "Switzerland",
  "metadata": {
    "source": "public-ai sovereignty pipeline",
    "version": "0.1.0",
    "uses_web": false,
    "requested_use_web": false,
    "uses_llm_web": false
  }
}
```

You can serialise this dict to JSON and keep it alongside your model, or use the values when editing the model card.

There is also a **separate organisation-level** helper:

```python
from pipeline.sovereignty_score import evaluate_organisation_for_hf

org_result = evaluate_organisation_for_hf("swiss-ai/Apertus-70B-2509")
print(org_result)
```

Example shape:

```json
{
  "model_id": "swiss-ai/Apertus-70B-2509",
  "organisation": "swiss-ai",
  "metric_name": "organisation_sovereignty_score",
  "metric_type": "custom",
  "value": 63.33,
  "dimensions": {
    "public_funding_score": 0.8,
    "independence_from_bigtech_score": 0.3,
    "local_sovereignty_focus_score": 0.8
  },
  "metadata": {
    "source": "public-ai organisation sovereignty heuristic",
    "version": "0.1.0"
  }
}
```

This keeps the **organisation sovereignty** signal clearly separated from the **model sovereignty** signal.

---

### Adding the result to a Hugging Face model card

In your model card `README.md` on Hugging Face, you can embed the score using the standard `model-index` block, for example:

```yaml
model-index:
- name: Apertus Sovereignty Evaluation
  results:
  - task:
      type: custom
      name: Sovereignty Assessment
    dataset:
      name: Web + Hub metadata
      type: custom
    metrics:
    - name: Sovereignty Score
      type: custom
      value: 62.5
    - name: Is the training data private?
      value: 0.50
    - name: Is the model trained locally?
      value: 0.70
    - name: Does the company have full control of the model weights?
      value: 0.30
    - name: Is the model trained on a different model?
      value: 0.80
    - name: Is the model weights private?
      value: 0.80
    - name: Is there country specific knowledge?
      value: 0.65
```

You would plug in the actual numbers produced by `evaluate_model_for_hf` for your model.

---

### Command‑line usage

There is also a simple CLI wrapper in `sovereignty_score.py`:

```bash
# Hub‑only (fast, metadata based)
python sovereignty_score.py "Apertus"

# With web evidence (needs ddgs + trafilatura)
python sovereignty_score.py "Apertus" --web

# JSON output (useful for piping into tools)
python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --json
```

This script uses the same underlying `compute_sovereignty_score` logic as the API.

---

### Dashboard dataset schema (`data/models.json`)

The GitHub Pages dashboard does not read Hugging Face directly. Instead, it loads a precomputed JSON snapshot at `data/models.json`, generated by `generate_data.py`.

`data/models.json` is an array of entries shaped like:

```json
[
  {
    "model": "Apertus-70B-Instruct-2509",
    "score": {
      "model_id": "swiss-ai/Apertus-70B-Instruct-2509",
      "metric_name": "sovereignty_score",
      "metric_type": "custom",
      "value": 62.5,
      "categories": {
        "Is the training data private?": 0.5,
        "Is the model trained locally?": 0.7,
        "Does the company have full control of the model weights?": 0.3,
        "Is the model trained on a different model?": 0.8,
        "Is the model weights private?": 0.8,
        "Is there country specific knowledge?": 0.65
      },
      "sources": [
        "https://arxiv.org/abs/....",
        "https://huggingface.co/...."
      ],
      "organisation_type": "State-backed",
      "country": "Switzerland",
      "metadata": {
        "source": "public-ai sovereignty pipeline",
        "version": "0.1.0",
        "uses_web": true,
        "requested_use_web": true,
        "uses_llm_web": false
      }
    },
    "explain": "Human-readable paragraph explaining how the final score was derived."
  }
]
```

Frontend mapping (what the website uses):
- Score chips use `score.value`
- Dimension bars use `score.categories`
- “How the score was derived” uses the `explain` paragraph
- “Scraped evidence sources” uses `score.sources`

---

### Generating `data/models.json`

From the repo root:

```bash
python generate_data.py
```

This runs the HF + (optional) web evidence pipeline and writes `data/models.json` for the dashboard.

---

### Methodology (short version)

- **Inputs**
  - Hugging Face model card metadata (license, tags, author/org, base model, cardData)
  - Optional web pages about the model, discovered via DuckDuckGo and scraped with `trafilatura`
  - Optional LLM scoring over those web pages (using the Public AI API, if configured)

- **Signals**
  - **License openness** → how private/public the weights appear to be
  - **Base model / fine‑tuning** → whether the model is trained from scratch or built on another model
  - **Organisation hints** → public / state‑backed orgs are treated as more sovereignty‑oriented by default
  - **Textual hints** (when web is enabled) such as “sovereign”, “sovereignty”, “open data”, “training data”, “transparent”

- **Aggregation**
  - Derive six category scores in \[0, 1\]
  - Compute the overall score as the **weighted average** of category scores, scaled to \[0, 100\]
  - Optionally blend in LLM‑based scores from web docs (50% Hub, 50% web) if enabled

---

### Assumptions and limitations

- This is a **heuristic** based on publicly available metadata and text, not a legal opinion.
- Signals and weights are opinionated and may not reflect your definition of “sovereignty”.
- Web‑based and LLM‑based components are **best‑effort** and can fail silently; when they do, the score falls back to Hub‑only signals.
- Always read the underlying model card and documentation; do not treat this as a definitive label.

