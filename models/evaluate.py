"""
EVALUATION HARNESS
==================

PURPOSE:
    Provide a consistent, fair way to compare every model —
    from the trivial random pricer to the fine-tuned LLM.

WHY A SHARED EVALUATOR:
    If each model uses different test data or different metrics,
    comparisons are meaningless. Every model in this project
    runs through the same function on the same test items.

METRIC: Mean Absolute Error (MAE)
    MAE = average of |predicted_price - actual_price|
    
    WHY MAE not MSE (Mean Squared Error):
        MAE is directly interpretable: "on average, our model
        is off by $58.74". MSE penalizes large errors more
        heavily, which would dominate our metric for a few
        wildly wrong predictions.
    
    WHY NOT RMSE:
        Same issue — dominated by outliers. We care about
        average performance across all products, not
        worst-case performance.
"""

import re
import random
from typing import Callable


def parse_price_from_output(output) -> float:
    """
    Extract a numeric price from a model's output.
    
    Different models return prices differently:
    - Traditional ML: float like 49.99
    - Fine-tuned LLM: string like "$49.99" or "49.99"
    - Zero-shot LLM:  string like "The price is approximately $50"
    
    We handle all cases with regex so the evaluator works
    for every model type without modification.
    
    Args:
        output: Model output (float, int, or string)
        
    Returns:
        Float price, or 0.0 if nothing parseable found
    """
    if isinstance(output, (int, float)):
        return max(0.0, float(output))
    
    if isinstance(output, str):
        # Remove currency symbols and formatting
        cleaned = output.replace('$', '').replace(',', '').strip()
        # Find first number pattern
        match = re.search(r'[-+]?\d*\.?\d+', cleaned)
        if match:
            return max(0.0, float(match.group()))
    
    return 0.0  # couldn't parse — return 0 as fallback


def evaluate_model(
    model_name:   str,
    predict_fn:   Callable,
    test_products: list,
    sample_size:  int = 200,
    random_seed:  int = 42
) -> dict:
    """
    Evaluate one pricing model on a random sample of test products.
    
    WHY sample_size=200 (not all 10,000 test items):
        - LLM inference costs money per API call
        - 200 samples gives statistically reliable results
          (confidence interval ± ~$5-10 for typical MAEs)
        - Allows fair comparison between cheap and expensive models
    
    WHY random_seed=42:
        Every model is evaluated on the SAME 200 items.
        Without this, model A might get easier items than model B.
    
    Args:
        model_name:    Display name for this model
        predict_fn:    Function(product) → price
        test_products: Full test set (we sample from this)
        sample_size:   How many items to evaluate on
        random_seed:   Ensures same items across models
        
    Returns:
        Dict with mae, errors, predictions, actuals
    """
    random.seed(random_seed)
    
    # Select the same random sample for every model
    size    = min(sample_size, len(test_products))
    indices = random.sample(range(len(test_products)), size)
    
    errors      = []
    predictions = []
    actuals     = []
    
    for i in indices:
        product = test_products[i]
        actual  = product.price
        
        # Get model prediction
        try:
            raw_output = predict_fn(product)
            predicted  = parse_price_from_output(raw_output)
        except Exception as e:
            print(f"  Warning: prediction failed for item {i}: {e}")
            predicted = 0.0
        
        error = abs(predicted - actual)
        errors.append(error)
        predictions.append(predicted)
        actuals.append(actual)
    
    mae = sum(errors) / len(errors)
    
    print(f"  {model_name:<40} MAE: ${mae:.2f}")
    
    return {
        'model_name':  model_name,
        'mae':         mae,
        'errors':      errors,
        'predictions': predictions,
        'actuals':     actuals,
        'sample_size': size,
    }


def print_results_table(all_results: list):
    """
    Print a formatted comparison table of all model results.
    
    Sorted from worst (highest MAE) to best (lowest MAE).
    Shows improvement over the random baseline for context.
    """
    sorted_results = sorted(all_results, key=lambda x: x['mae'], reverse=True)
    baseline_mae   = sorted_results[0]['mae']  # worst = random
    
    print("\n" + "="*65)
    print("  MODEL COMPARISON RESULTS")
    print("="*65)
    print(f"  {'Model':<40} {'MAE':>8}  {'vs Random':>10}")
    print("-"*65)
    
    for r in sorted_results:
        improvement = baseline_mae - r['mae']
        bar = '█' * int(r['mae'] / 15)
        print(f"  {r['model_name']:<40} ${r['mae']:>7.2f}  "
              f"{'-' if improvement == 0 else f'-${improvement:.0f}':>10}  {bar}")
    
    print("="*65)
    best = sorted_results[-1]
    print(f"\n  Best model: {best['model_name']}")
    print(f"  Best MAE:   ${best['mae']:.2f}")
    print(f"  Improvement over random: ${baseline_mae - best['mae']:.2f} "
          f"({100*(baseline_mae - best['mae'])/baseline_mae:.1f}%)")
