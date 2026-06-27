# Amazon Price Predictor

I built this project to learn how LLM engineering actually works end to end — from raw data all the way to a deployed model. The idea was simple: given a product description, can a model predict its price? Turns out it can, and a fine-tuned LLM does it better than traditional ML.

---

## What I Was Trying To Learn

I wanted to understand the full pipeline — not just train a model and call it done. So I went through everything:

- How do you curate a large dataset at scale?
- Does text preprocessing with an LLM actually help downstream models?
- How much better is a fine-tuned LLM compared to XGBoost on a text-based task?
- How do you actually deploy a model so it runs as a real API?
- What does a multi-agent system look like in practice?

This project answered all of those for me.

---

## The Result

| Model | MAE ($) |
|-------|---------|
| Random guess | ~$287 |
| Always predict mean | ~$141 |
| Linear Regression (3 features) | ~$120 |
| Linear Regression (Bag of Words) | ~$100 |
| Random Forest | ~$85 |
| XGBoost | $68.23 |
| **Fine-tuned Llama 3.2-3B** | **$58.74** |

The fine-tuned LLM is 13.9% better than XGBoost. The reason is simple — XGBoost counts words independently. The LLM understands what those words mean together. "Sony noise-cancelling headphones" isn't three separate words to an LLM, it's a premium product category.

---

## The Data

Started with the McAuley-Lab Amazon Reviews 2023 dataset — about 29 lakh products across 8 categories.

**The pipeline:**
```
29 lakh raw items
      ↓  filter: price $0.50–$999, min 300 chars text, remove part numbers
~20 lakh clean items
      ↓  deduplicate: remove same title, same description
~18 lakh unique items
      ↓  weighted sampling: bias toward expensive items,
         down-weight Automotive (was 33% of data alone)
8.2 lakh balanced items
      ↓  split 800k train / 10k val / 10k test
pushed to huggingface.co/datasets/maulik78/items_raw_full
```

One thing I learned here — imbalanced data is a real problem. Automotive had 9.7 lakh items on its own. If I'd sampled uniformly the model would've basically been an Automotive price predictor. Weighted sampling fixed that.

---

## LLM Preprocessing

Raw Amazon descriptions are messy. Part numbers, HTML artifacts, inconsistent formatting, 4000 characters of marketing copy. I used Groq's batch API to rewrite every product into a clean structured format:

```
Before:
"B07XJ8C8F5 Schlage F59 AND 613 Andover Interior Knob...
 [From the Manufacturer] When you have a Schlage handleset..."
 
After:
Title: Schlage Andover Interior Door Knob with Deadbolt
Category: Home Hardware
Brand: Schlage
Description: Oil-rubbed bronze interior knob for secure door use.
Details: Requires 4-inch door prep, lifetime warranty.
```

This made a noticeable difference for all the models downstream.

Preprocessed dataset: [maulik78/items_full](https://huggingface.co/datasets/maulik78/items_full)

---

## Traditional ML Baselines

Before touching an LLM I ran 6 baseline models. This was important — without baselines I'd have no idea if the LLM was actually useful or just overengineered.

The progression from random guess → XGBoost showed me exactly how much each technique adds. XGBoost at $68 MAE became the benchmark everything else had to beat.

One interesting finding from XGBoost's feature importance — the words that predicted price most strongly were brand names and quality indicators like "professional" and "commercial". Not product type. Brand + quality tier matters more than what the product actually is.

---

## Fine-tuning with QLoRA

Fine-tuning Llama 3.2-3B normally needs ~28GB VRAM. I did it on a free Colab T4 (15GB) using QLoRA.

**How QLoRA makes this possible:**

Quantization compresses the model from 32-bit to 4-bit NF4 format:
```
6.4 GB → 2.2 GB
```

LoRA trains only small adapter matrices instead of all 3 billion weights:
```
W_new = W_frozen + alpha × (A × B)
A = (3072 × 32),  B = (32 × 3072)
3B trainable params → 18M trainable params
```

The adapters are injected into the attention layers — the parts of the model that decide what to focus on. For pricing, I wanted the model to focus more on brand and quality signals.

**Training setup:**
- Dataset: 20,000 items (lite mode, free T4)
- Batch size: 4, gradient accumulation: 8 (effective batch = 32)
- Learning rate: 1e-4 with cosine decay
- Time: 1 hour 54 minutes

**Training progress:**
| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 100 | 1.283 | 1.280 |
| 300 | 1.235 | 1.258 |
| 500 | 1.240 | 1.250 |
| 625 | 1.265 | 1.248 |

Val loss kept decreasing throughout — no overfitting on 20k items with 1 epoch.

**Result: $58.74 MAE**

Model: [maulik78/pricer-2026-06-10_06.40.40-lite](https://huggingface.co/maulik78/pricer-2026-06-10_06.40.40-lite)  
W&B run: [wandb.ai/maulik04-lnmiit/pricer](https://wandb.ai/maulik04-lnmiit/pricer)

---

## Deployment on Modal

After training I wanted the model to actually be usable, not just sitting in a notebook. I deployed it on Modal — a serverless GPU platform.

The key thing I learned here: use a class-based deployment, not function-based. With a class, the model loads once when the container starts and stays in memory for all subsequent calls. With a function it can reload every call — which would mean 30 seconds per prediction.

```python
@app.cls(gpu="T4")
class Pricer:
    @modal.enter()
    def setup(self):
        # runs once — load model here
        self.model = load_model()
    
    @modal.method()
    def price(self, description):
        # runs per request — model already loaded
        return predict(self.model, description)
```

Cold start (first call): ~30-60 seconds  
Warm calls: ~2-3 seconds

Deployment: [modal.com/apps/maulikalwar04/main/deployed/pricer-service](https://modal.com/apps/maulikalwar04/main/deployed/pricer-service)



## Datasets on HuggingFace

| Dataset | Items | What's in it |
|---------|-------|--------------|
| [maulik78/items_raw_full](https://huggingface.co/datasets/maulik78/items_raw_full) | 820k | Raw parsed products |
| [maulik78/items_raw_lite](https://huggingface.co/datasets/maulik78/items_raw_lite) | 22k | Same, smaller |
| [maulik78/items_full](https://huggingface.co/datasets/maulik78/items_full) | 820k | LLM-preprocessed |
| [maulik78/items_lite](https://huggingface.co/datasets/maulik78/items_lite) | 22k | Same, smaller |
| [maulik78/items_prompts_full](https://huggingface.co/datasets/maulik78/items_prompts_full) | 820k | Training prompts |
| [maulik78/items_prompts_lite](https://huggingface.co/datasets/maulik78/items_prompts_lite) | 22k | Same, smaller |

---

## How To Run It

```bash
git clone https://github.com/maulik-04/amazon-price-predictor.git
cd amazon-price-predictor
pip install -r requirements.txt
cp .env.example .env
# add your API keys to .env
```

**Get a price prediction:**
```python
from deployment.predict import RemotePredictor

predictor = RemotePredictor()
price = predictor.predict("""
Title: Sony WH-1000XM4 Wireless Headphones
Category: Electronics
Brand: Sony
Description: Noise cancelling wireless headphones
Details: 30 hour battery, touch controls
""")
print(f"${price:.2f}")
```

**Run the Streamlit demo:**
```bash
streamlit run app/streamlit_app.py
```

**Deploy to Modal:**
```bash
modal setup
modal secret create huggingface HF_TOKEN=your_token
modal deploy deployment/modal_service.py
```

---



*Maulik Mathur · B.Tech CSE (AI & DS) · LNMIIT Jaipur*
