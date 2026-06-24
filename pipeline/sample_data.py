"""
STEP 3: Weighted Sampling for a Balanced Dataset
==================================================

PURPOSE:
    We have ~2.9 million clean items but need a balanced,
    representative sample of ~820,000 for training.

THE IMBALANCE PROBLEM:
    Raw counts after cleaning:
    - Automotive:                 974,000 items (33% of total!)
    - Tools_and_Home_Improvement: 552,000 items (19%)
    - Electronics:                464,000 items (16%)
    - Toys_and_Games:             348,000 items (12%)
    ...
    - Appliances:                  35,000 items (1.2%)
    - Musical_Instruments:         69,000 items (2.4%)
    
    If we sample uniformly, our model would be 33% Automotive.
    It would learn Automotive pricing very well but generalize
    poorly to other categories.

THE PRICE IMBALANCE PROBLEM:
    Most products are cheap ($5-$30). Very few are expensive
    ($500-$999). But high-price items contain more signal —
    a $900 product has many distinguishing features in its
    description. If we oversample cheap items, the model
    learns "when in doubt, predict $30."

OUR SOLUTION: Two-dimensional weighted sampling
    1. Bias toward expensive items (squaring normalized price)
    2. Down-weight dominant categories (Automotive × 0.05)

WHY SQUARING THE PRICE WEIGHT:
    If we just normalize price to [0,1], a $100 item gets weight
    0.1 and a $900 item gets weight 0.9. After squaring:
    $100 → 0.01, $900 → 0.81
    This much more aggressively promotes expensive items.
"""

import numpy as np
from collections import Counter


# ── Sampling Configuration ─────────────────────────────────────
TARGET_SAMPLE_SIZE = 820_000

# Category multipliers — applied AFTER price weighting
# These correct for category imbalance
# Values < 1.0 reduce a category's representation
# Values = 1.0 leave the category at its natural weight
CATEGORY_MULTIPLIERS = {
    'Automotive':                  0.05,  # 974k → ~48k (92k if 0.1)
    'Tools_and_Home_Improvement':  0.50,  # 552k → ~188k
    'Electronics':                 1.00,  # keep full representation
    'Toys_and_Games':              1.00,
    'Office_Products':             1.00,
    'Cell_Phones_and_Accessories': 1.00,
    'Musical_Instruments':         1.00,
    'Appliances':                  1.00,  # only 35k, keep all
}


def compute_sampling_weights(products: list) -> np.ndarray:
    """
    Compute a sampling probability for each product.
    
    FORMULA:
        normalized_price = (price - min_price) / (max_price - min_price)
        base_weight      = normalized_price ** 2
        final_weight     = base_weight × category_multiplier
        probability      = final_weight / sum(all_weights)
    
    The final probability array sums to 1.0 — required by
    np.random.choice when using the 'p' parameter.
    
    Args:
        products: List of Product objects
        
    Returns:
        Array of probabilities, one per product, summing to 1.0
    """
    prices     = np.array([p.price    for p in products], dtype=float)
    categories = np.array([p.category for p in products], dtype=str)
    
    # Step 1: Normalize prices to [0, 1]
    # Small epsilon (1e-9) prevents division by zero if all prices are equal
    price_range      = prices.max() - prices.min()
    normalized_prices = (prices - prices.min()) / (price_range + 1e-9)
    
    # Step 2: Square to bias toward expensive items
    # Effect: $50 item (normalized 0.05) → weight 0.0025
    #         $900 item (normalized 0.9)  → weight 0.81
    weights = normalized_prices ** 2
    
    # Step 3: Apply category multipliers to correct imbalance
    for category, multiplier in CATEGORY_MULTIPLIERS.items():
        mask          = categories == category
        items_in_cat  = mask.sum()
        before        = weights[mask].sum()
        weights[mask] *= multiplier
        after         = weights[mask].sum()
        
        if items_in_cat > 0:
            print(f"  {category:<35} ×{multiplier} "
                  f"({items_in_cat:,} items, weight {before:.1f} → {after:.1f})")
    
    # Step 4: Normalize so probabilities sum to exactly 1.0
    # Required by np.random.choice
    weights = weights / weights.sum()
    
    return weights


def sample_products(products: list, random_seed: int = 42) -> list:
    """
    Sample TARGET_SAMPLE_SIZE products using computed weights.
    
    WHY replace=False:
        We don't want the same product appearing twice in our
        training data. Duplicates would cause the model to
        memorize specific products rather than learning patterns.
    
    WHY seed=42:
        Makes the sampling reproducible. Anyone running this
        code gets the exact same sample, enabling fair comparisons.
    
    Args:
        products:    All cleaned products from all categories
        random_seed: For reproducibility
        
    Returns:
        Sampled and shuffled list of products
    """
    np.random.seed(random_seed)
    
    sample_size = min(TARGET_SAMPLE_SIZE, len(products))
    
    print(f"\nComputing sampling weights for {len(products):,} products...")
    weights = compute_sampling_weights(products)
    
    print(f"\nSampling {sample_size:,} products...")
    selected_indices = np.random.choice(
        len(products),    # sample from this many items
        size=sample_size, # how many to select
        replace=False,    # no duplicates
        p=weights         # probability of selecting each item
    )
    
    sampled = [products[i] for i in selected_indices]
    
    # Shuffle the result so category ordering is mixed
    # Important: without this, all Automotive items would be together
    import random
    random.seed(random_seed)
    random.shuffle(sampled)
    
    # Report final distribution
    category_counts = Counter(p.category for p in sampled)
    print(f"\nFinal sample distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(sampled) * 100
        bar = '█' * int(pct / 2)
        print(f"  {cat:<35} {count:>8,}  ({pct:.1f}%) {bar}")
    
    return sampled


def split_dataset(products: list) -> tuple:
    """
    Split products into train, validation, and test sets.
    
    SPLIT RATIOS:
        Train: 97.56% (800k of 820k)
        Val:   1.22%  (10k  of 820k)
        Test:  1.22%  (10k  of 820k)
    
    WHY THESE RATIOS:
        Test set needs to be large enough for stable evaluation
        (10k gives narrow confidence intervals on MAE).
        Val set monitors for overfitting during training.
        Train gets the rest — more is always better for ML.
    
    WHY WE SPLIT AFTER SAMPLING:
        We sample first to get the right distribution,
        then split so train/val/test have similar distributions.
        If we split first and sampled only train, val and test
        would have different category distributions.
    
    Returns:
        (train, val, test) tuples
    """
    n = len(products)
    
    if n >= 820_000:
        train = products[:800_000]
        val   = products[800_000:810_000]
        test  = products[810_000:820_000]
    else:
        # Proportional split for smaller datasets
        train_end = int(n * 0.9756)
        val_end   = int(n * 0.9878)
        train = products[:train_end]
        val   = products[train_end:val_end]
        test  = products[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train):,} ({100*len(train)/n:.1f}%)")
    print(f"  Val:   {len(val):,}   ({100*len(val)/n:.1f}%)")
    print(f"  Test:  {len(test):,}  ({100*len(test)/n:.1f}%)")
    
    return train, val, test
