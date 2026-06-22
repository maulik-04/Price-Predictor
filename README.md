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
