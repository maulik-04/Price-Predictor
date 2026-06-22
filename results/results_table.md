# Model Results

## Evaluation Setup
- **Test Set:** 10,000 held-out Amazon products
- **Evaluation Metric:** Mean Absolute Error (MAE) in USD
- **Evaluation Size:** 200 random test items per model
- **Dataset:** maulik78/items_full (800k train / 10k val / 10k test)
- **Price Range:** $0.50 – $999.49

## Results

| Rank | Model | MAE ($) | vs XGBoost | Notes |
|------|-------|---------|------------|-------|
| - | Random Pricer | ~$287 | +$219 | Floor baseline |
| - | Constant Pricer | ~$141 | +$73 | Always predicts mean |
| - | Linear Regression (3 feat) | ~$120 | +$52 | Weight + text length |
| - | NL Linear Regression | ~$100 | +$32 | 2,000 BoW features |
| - | Random Forest (15k) | ~$85 | +$17 | 100 decision trees |
| 2 | XGBoost (800k) | $68.23 | baseline | Gradient boosting |
| 1 | **Fine-tuned Llama 3.2-3B** | **$58.74** | **-$9.49 (13.9%)** | QLoRA, 20k items |


## Fine-tuning Training Progress

| Step | Train Loss | Val Loss | Token Accuracy |
|------|-----------|----------|----------------|
| 100 | 1.283 | 1.280 | 76.1% |
| 200 | 1.249 | 1.273 | 75.9% |
| 300 | 1.235 | 1.258 | 76.0% |
| 400 | 1.265 | 1.257 | 76.3% |
| 500 | 1.240 | 1.250 | 76.3% |
| 600 | 1.308 | 1.248 | 76.2% |
| 625 | 1.265 | 1.248 | 76.2% |

## Sample Predictions (Fine-tuned Model)

| Product | Actual Price | Predicted | Error |
|---------|-------------|-----------|-------|
| Excess V2 Distortion Pedal | $219.00 | $200.00 | $19.00 |
| Telpo Headlight Assembly | $115.99 | $100.00 | $15.99 |
| NewPower99 Battery Kit | $54.95 | $20.00 | $34.95 |

## Key Insights

1. **LLM beats BoW:** Fine-tuned LLM understands semantic context
   (Sony + noise-cancelling = premium) while XGBoost sees independent word counts

2. **Data quality matters:** LLM preprocessing (converting messy Amazon
   descriptions to clean summaries) was critical for all downstream models

3. **Lite mode limitation:** Training on 20k items causes the model to
   default toward mean prices for unseen product types. Full 800k training
   resolves this.

4. **QLoRA efficiency:** 18M trainable parameters (0.6% of 3B) on a free
   T4 GPU in under 2 hours — demonstrates parameter-efficient fine-tuning

## W&B Training Run
https://wandb.ai/maulik04-lnmiit/pricer/runs/fj0m46rt
