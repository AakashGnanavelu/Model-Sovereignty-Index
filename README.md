## Public AI Sovereignty Score

View Dashboard from this [link](https://aakashgnanavelu.github.io/Model-Sovereignty-Index/).

This repo contains a pipeline that estimates a **model sovereignty score** using:

- **Hugging Face Hub metadata** (primary source)
- Optional **web evidence** (search + scraping)
- Optional **LLM reasoning over web evidence**

The goal is to provide a **simple, reproducible heuristic** that can be used as a **custom evaluation metric** on Hugging Face model cards — not a definitive judgement of legal, geopolitical, or organisational sovereignty.

---

## What is the sovereignty score?

For a given model, we compute:

- **Overall score**: `0–100`
- **Per-category scores**: `0–1`

### Categories

- `Is the training data private?`
- `Is the model trained locally?`
- `Does the company have full control of the model weights?`
- `Is the model trained on a different model?`
- `Is there country specific knowledge?`

> ⚠️ Note: Earlier versions duplicated “Is the model weights private?” — this has been removed and merged into the correct categories.

---

## How the score is derived

### 1. Hugging Face (primary signal)

We extract structured metadata from the Hugging Face API:

- `license`
- `author / organisation`
- `tags`
- `cardData` (including `base_model`)

From this, we compute heuristics:

| Signal | Effect |
|--------|--------|
| Open license | ↓ “private weights”, ↑ control |
| Base model present | ↓ sovereignty (dependent model) |
| No base model | ↑ sovereignty (trained from scratch) |
| Public/state org hints | ↑ local training + country relevance |
| Transparency / open tags | ↓ “training data private” |

This produces a **baseline score per category**.

---

### 2. Organisation & country inference

We improve accuracy by querying:

- Hugging Face **organisation API**
- Org metadata (description, blog, GitHub)
- Domain inference (e.g. `.fr`, `.ch`)

This is significantly more reliable than keyword matching alone.

---

### 3. Optional: Web evidence

If `--web` is enabled:

- Search for model-related pages
- Extract content using `trafilatura`
- Filter by category relevance

This provides **supporting evidence**, not raw scoring.

---

### 4. Optional: LLM scoring over web (`--llm`)

If enabled:

- Each category is scored independently using an LLM
- Quotes are:
  - Verified against sources
  - Filtered for relevance and quality

Final score: $$0.5 * \text{Hugging Face score} + 0.5 * \text{Web (LLM) score}$$


---

### 5. Final aggregation

- Each category ∈ `[0,1]`
- Weighted average (equal weights by default)
- Scaled to `[0,100]`

---

## Programmatic API

### Model evaluation

```python
from pipeline.sovereignty_score import evaluate_model_for_hf
from pipeline.sovereignty_score import explain_sovereignty_score

result = evaluate_model_for_hf(
    "swiss-ai/Apertus-70B-2509",
    use_web=False,
    use_llm_web=False,
)
explanation = explain_sovereignty_score(result)
```

This produces a human-readable explanation grounded in the actual scores and evidence.

## CLI Usage

```bash
# Basic score (fast, HF only)
python sovereignty_score.py "swiss-ai/Apertus-70B-2509"

# With web evidence
python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --web

# With LLM scoring over web
python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --web --llm

# Explanation (prints score + explanation)
python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --explain

# Full JSON output
python sovereignty_score.py "swiss-ai/Apertus-70B-2509" --json
```
### Important behaviour

- `--explain` → prints:
  - score
  - explanation
- `--json` → prints:
  - full structured output (no explanation)
- If both are passed:
  - JSON is returned (machine-readable takes priority)

---

### Example output

```json
{
  "model_id": "swiss-ai/Apertus-70B-2509",
  "value": 62.5,
  "categories": {
    "Is the training data private?": 0.5,
    "Is the model trained locally?": 0.7,
    "Does the company have full control of the model weights?": 0.3,
    "Is the model trained on a different model?": 0.8,
    "Is there country specific knowledge?": 0.65
  },
  "organisation_type": "State-backed",
  "country": "Switzerland"
}
```

## Dashboard dataset (data/models.json)

## Dashboard dataset (`data/models.json`)

The dashboard uses precomputed results:

- `score.value` → overall score  
- `score.categories` → per-dimension bars  
- `explain` → explanation text  
- `score.sources` → evidence links  

Generated via:

```bash
python generate_data.py
```

## Sovereignty ≈ control, independence, transparency, locality

---

### Pipeline

- Extract structured metadata  
- Apply heuristics → category scores  
- (Optional) refine with web + LLM  
- Aggregate → final score  

---

### Assumptions and limitations

- This is a heuristic, not a legal or political classification  
- Hugging Face metadata can be:  
  - incomplete  
  - inconsistent  
  - missing key facts  
- Web scraping:  
  - may fail  
  - may retrieve irrelevant content  
- LLM scoring:  
  - depends on prompt quality  
  - may introduce bias despite filtering  
- Organisation classification is approximate  
- “Sovereignty” is subjective and context-dependent  

---

### Design philosophy

This project prioritises:

- **Transparency** → clear scoring logic  
- **Reproducibility** → same inputs → same outputs  
- **Extensibility** → easy to add signals or categories  
- **Practicality** → usable directly in model cards  

---

### When to use this

#### Useful for:

- Comparing models at a high level  
- Adding structured evaluation to model cards  
- Exploring trends (open vs closed, local vs global)  

#### Not suitable for:

- Legal compliance decisions  
- Security guarantees  
- Political classification  