# 🏷️ AI Price Predictor

> Predict the price of any Amazon product from its text description using a fine-tuned Llama 3.2-3B model deployed on Modal cloud infrastructure.

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-maulik78%2Fpricer-blue)](https://huggingface.co/maulik78/pricer-2026-06-10_06.40.40-lite)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-maulik78%2Fitems__full-green)](https://huggingface.co/datasets/maulik78/items_full)
[![Modal Deployment](https://img.shields.io/badge/Modal-Deployed-purple)](https://modal.com/apps/maulikalwar04/main/deployed/pricer-service)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://python.org)

---

## 📌 Project Overview

This project investigates whether modern Large Language Models can understand
product attributes, brands, and features well enough to predict prices — and
whether fine-tuning outperforms traditional ML approaches.

**Answer: Yes. Fine-tuned Llama 3.2-3B achieves $58.74 MAE, beating XGBoost
($68.23 MAE) by 13.9% — trained on just 20,000 items using a free T4 GPU.**

### What This Project Covers

| Phase | Description | Key Technology |
|-------|-------------|----------------|
| Data Engineering | Curate 820k Amazon products | HuggingFace Datasets, Pydantic |
| LLM Preprocessing | Standardize noisy descriptions | Groq Batch API, JSONL |
| Traditional ML | 6 baseline models | scikit-learn, XGBoost |
| LLM Fine-tuning | QLoRA on Llama 3.2-3B | PEFT, TRL, bitsandbytes |
| Deployment | Serverless GPU API | Modal, HuggingFace Hub |
| Multi-Agent System | Automated deal detection | LiteLLM, ChromaDB, Pushover |

---

## 📊 Results

| Model | MAE ($) | Notes |
|-------|---------|-------|
| Random Pricer | ~$287 | Floor baseline |
| Constant Pricer (mean) | ~$141 | Always predicts average |
| Linear Regression (3 features) | ~$120 | Weight + text length |
| NL Linear Regression (BoW) | ~$100 | 2,000 word features |
| Random Forest (15k subset) | ~$85 | 100 decision trees |
| XGBoost (800k dataset) | $68.23 | Gradient boosting |
| **Fine-tuned Llama 3.2-3B (lite)** | **$58.74** | **QLoRA, 20k items, T4 GPU** |
| Reference — Full model | ~$52 | 800k items, A100 (instructor) |

---

## 🏗️ Architecture

Raw Amazon Data (48M products, 33 categories)

↓

Data Curation (filter, parse, deduplicate, weighted sample)

↓

LLM Preprocessing (Groq batch API → clean summaries)

↓

Traditional ML Baselines (Linear Regression → XGBoost)

↓

QLoRA Fine-tuning (Llama 3.2-3B, T4 GPU, 1h 54min)

↓

Modal Deployment (serverless GPU API)

↓

Multi-Agent System

├── SpecialistAgent  → Fine-tuned Llama via Modal

├── FrontierAgent    → Groq Llama 70B + RAG (ChromaDB)

├── EnsembleAgent    → Weighted average (70/30)

├── ScannerAgent     → RSS deal scraping + LLM filtering

└── MessagingAgent   → Pushover push notifications


---

## 🗂️ HuggingFace Datasets

All datasets are publicly available on HuggingFace:

| Dataset | Size | Description |
|---------|------|-------------|
| [maulik78/items_raw_full](https://huggingface.co/datasets/maulik78/items_raw_full) | 820k items | Raw parsed Amazon products |
| [maulik78/items_raw_lite](https://huggingface.co/datasets/maulik78/items_raw_lite) | 22k items | Lite version for experimentation |
| [maulik78/items_full](https://huggingface.co/datasets/maulik78/items_full) | 820k items | LLM-preprocessed clean summaries |
| [maulik78/items_lite](https://huggingface.co/datasets/maulik78/items_lite) | 22k items | Lite preprocessed version |
| [maulik78/items_prompts_full](https://huggingface.co/datasets/maulik78/items_prompts_full) | 820k items | Training prompts for fine-tuning |
| [maulik78/items_prompts_lite](https://huggingface.co/datasets/maulik78/items_prompts_lite) | 22k items | Lite prompts version |

---

## 🤖 Fine-tuned Model

**Model:** [maulik78/pricer-2026-06-10_06.40.40-lite](https://huggingface.co/maulik78/pricer-2026-06-10_06.40.40-lite)

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-3.2-3B |
| Training Data | 20,000 Amazon products |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Trainable Parameters | 18M / 3B (0.6%) |
| Quantization | 4-bit NF4 |
| Training Time | 1h 54min on T4 GPU |
| Epochs | 1 |
| Final Val Loss | 1.248 |
| MAE on Test Set | $58.74 |

### Training Curves

| Step | Train Loss | Val Loss | Accuracy |
|------|-----------|----------|----------|
| 100 | 1.283 | 1.280 | 76.1% |
| 200 | 1.249 | 1.273 | 75.9% |
| 300 | 1.235 | 1.258 | 76.0% |
| 400 | 1.265 | 1.257 | 76.3% |
| 500 | 1.240 | 1.250 | 76.3% |
| 600 | 1.308 | 1.248 | 76.2% |
| 625 | 1.265 | 1.248 | 76.2% |

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/maulik-04/ai-price-predictor.git
cd ai-price-predictor
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run a Prediction (Local)

```python
from agents.specialist_agent import SpecialistAgent

agent = SpecialistAgent()
price = agent.price("""
Title: Sony WH-1000XM4 Wireless Headphones
Category: Electronics
Brand: Sony
Description: Industry leading noise cancelling headphones
Details: 30 hour battery, touch controls, Alexa built in
""")
print(f"Predicted price: ${price:.2f}")
```

### 4. Run Full Deal Detection Pipeline

```python
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent

scanner   = ScannerAgent()
ensemble  = EnsembleAgent()
messenger = MessagingAgent()

deals = scanner.scan()
for deal in deals:
    estimate = ensemble.price(deal.description)
    discount = (estimate - deal.price) / estimate
    if discount >= 0.4:
        messenger.notify(deal.title, deal.price, estimate, deal.url)
```

---

## 📁 Repository Structure

ai-price-predictor/

├── README.md                    ← You are here

├── requirements.txt             ← Dependencies

├── .env.example                 ← API key template

├── .gitignore

│

├── pricer/                      ← Core data pipeline

│   ├── items.py                 ← Item Pydantic model

│   ├── parser.py                ← Raw data parser

│   ├── loaders.py               ← Category loader

│   ├── evaluator.py             ← Evaluation harness

│   ├── batch.py                 ← Groq batch processor

│   └── deep_neural_network.py  ← PyTorch DNN

│

├── notebooks/

│   ├── 01_data_curation.ipynb

│   ├── 02_preprocessing.ipynb

│   ├── 03_baselines.ipynb

│   ├── 04_neural_network.ipynb

│   ├── 05_qlora_training.ipynb

│   └── 06_agents.ipynb

│

├── agents/                      ← Multi-agent system

│   ├── specialist_agent.py      ← Fine-tuned Llama via Modal

│   ├── frontier_agent.py        ← Groq Llama 70B + RAG

│   ├── ensemble_agent.py        ← Weighted combination

│   ├── scanner_agent.py         ← RSS deal scraper

│   └── messaging_agent.py       ← Pushover notifications

│

├── deployment/

│   ├── pricer_service.py        ← Modal serverless deployment

│   └── test_pricer.py           ← Deployment tests

│

└── app/

└── streamlit_app.py         ← Interactive web demo


---

## 🔧 Setup Guide

### Prerequisites

- Python 3.11+
- HuggingFace account (free)
- Groq account (free tier)
- Modal account (free $30 credits)
- Pushover account (optional, for notifications)

### API Keys Required

| Key | Where to Get | Required |
|-----|-------------|----------|
| `HF_TOKEN` | huggingface.co/settings/tokens | Yes |
| `GROQ_API_KEY` | console.groq.com | Yes |
| `OPENAI_API_KEY` | platform.openai.com | Optional |
| `PUSHOVER_USER` | pushover.net | Optional |
| `PUSHOVER_TOKEN` | pushover.net | Optional |

### Deploy to Modal

```bash
cd deployment
modal deploy pricer_service.py
```

---

## 📓 Notebooks Guide

| Notebook | Runtime | Time | Description |
|----------|---------|------|-------------|
| 01_data_curation | Colab CPU | 3-5 hrs | Load and curate 820k Amazon products |
| 02_preprocessing | Colab CPU | 30 min | Copy preprocessed data to your HF account |
| 03_baselines | Colab CPU | 1-2 hrs | Train and evaluate 6 ML models |
| 04_neural_network | Colab GPU T4 | 2-3 hrs | Train PyTorch deep neural network |
| 05_qlora_training | Colab GPU T4 | 2 hrs | QLoRA fine-tune Llama 3.2-3B |
| 06_agents | Local | 30 min | Run multi-agent pipeline |

---

## 🧠 Key Technical Concepts

### Why QLoRA?

Full fine-tuning of Llama 3.2-3B requires ~24GB VRAM. QLoRA reduces this to 2.3GB:

Quantization:  6.4GB (float32) → 2.2GB (4-bit NF4)

LoRA:          3B parameters → 18M trainable parameters

Result:        Fits on free T4 GPU

### Why Fine-tuned LLM Beats XGBoost

XGBoost sees words as independent features. The LLM understands context:

XGBoost: "Sony" + "noise" + "cancelling" → independent word counts

LLM:     "Sony WH-1000XM4 noise-cancelling" → premium consumer electronics

### Weighted Sampling Strategy

```python
# Normalize prices to [0,1]
p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

# Square to bias toward higher-priced items
w = p ** 2

# Down-weight dominant categories
w[categories == "Automotive"] *= 0.05
w[categories == "Tools_and_Home_Improvement"] *= 0.5

# Sample 820k items
idx = np.random.choice(len(items), size=820_000, replace=False, p=w)
```

---

## 📈 W&B Training Dashboard

Training metrics logged to Weights & Biases:
[wandb.ai/maulik04-lnmiit/pricer](https://wandb.ai/maulik04-lnmiit/pricer)

---

## 🙏 Acknowledgements

- Dataset: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) — Julian McAuley, UC San Diego
- Course: LLM Engineering by Ed Donner (Udemy) — project built as a guided capstone
- Base Model: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

---

## 📄 License

MIT License — see LICENSE file for details.
