"""
STEP 2: Clean and Parse Raw Product Data
=========================================

PURPOSE:
    Transform messy raw Amazon listings into clean structured
    records ready for machine learning.

THE CLEANING PROBLEM:
    Raw Amazon data contains:
    - Prices as strings: '$49.99', '$1,299.00', '' (missing)
    - Part numbers: 'Part# AX-7823-B' pollute text features
    - HTML artifacts in descriptions
    - Extremely short or extremely expensive items (noise/outliers)
    - Multiple units for weight (pounds, ounces, grams, kg)

OUR DECISIONS:
    1. Price range $0.50 - $999.49
       WHY: Under $0.50 are usually shipping accessories or samples.
       Over $999 are industrial/commercial items that behave differently
       from consumer products. We focus on the consumer price range.
    
    2. Minimum 300 characters of text
       WHY: Very short descriptions don't give enough signal for
       a language model to predict price meaningfully.
    
    3. Remove part numbers using regex
       WHY: Part numbers like 'B07XJ8C8F5' cause memorization —
       the model learns "this exact part number = $49" instead of
       understanding what the product IS.

OUTPUT:
    List of clean product dictionaries ready for Step 3 (sampling)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Price Filter Constants ────────────────────────────────────
MIN_PRICE     = 0.50    # ignore items below this price
MAX_PRICE     = 999.49  # ignore items above this price
MIN_TEXT_LEN  = 300     # ignore items with very short descriptions


# ── Part Number Pattern ───────────────────────────────────────
# Matches strings that are 7+ chars of uppercase letters + numbers
# Examples: 'B07XJ8C8F5', 'AX-7823-B', 'SKU-1234ABC'
# These are catalog IDs, not useful text features
PART_NUMBER_PATTERN = re.compile(
    r'\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b'
)


@dataclass
class Product:
    """
    A single cleaned product ready for ML training.
    
    We use a dataclass (not a complex class) so the structure
    is simple, transparent, and easy to serialize to JSON/parquet.
    
    WHY THESE FIELDS:
    - text: the main input to our ML models
    - price: the target we're trying to predict
    - category: useful for analysis and stratified sampling
    - weight: a numerical feature for traditional ML models
      (heavier products often cost more)
    - summary: added in Step 5 after LLM preprocessing
    """
    title:    str
    category: str
    price:    float
    text:     str               # concatenated title + description + features + details
    weight:   float = 0.0       # in pounds, 0.0 if unknown
    summary:  Optional[str] = None  # filled in by LLM preprocessing (Step 5)


def parse_price(price_str) -> Optional[float]:
    """
    Convert raw price string to float.
    
    Raw prices look like: '$49.99', '$1,299.00', '49.99', ''
    We return None for anything we can't parse — the caller
    decides what to do with unparseable prices (we skip them).
    """
    if not price_str:
        return None
    
    try:
        # Remove $ sign and commas, then convert to float
        cleaned = str(price_str).replace('$', '').replace(',', '').strip()
        price   = float(cleaned)
        return price
    except (ValueError, TypeError):
        return None


def extract_weight_in_pounds(details: dict) -> float:
    """
    Extract product weight and convert everything to pounds.
    
    WHY: We want a single consistent unit.
    WHY POUNDS: The original dataset uses pounds most often.
    
    Handles: pounds, ounces, grams, milligrams, kilograms
    Returns 0.0 if weight is missing or unparseable.
    """
    weight_str = details.get('Item Weight', '')
    
    if not weight_str:
        return 0.0
    
    try:
        parts  = str(weight_str).split()
        amount = float(parts[0])
        unit   = parts[1].lower() if len(parts) > 1 else ''
        
        # Convert everything to pounds
        conversions = {
            'pounds': 1.0,
            'ounces': 1/16,
            'grams':  1/453.592,
            'milligrams': 1/453592,
            'kilograms':  1/0.453592,
        }
        
        multiplier = conversions.get(unit, 0.0)
        return round(amount * multiplier, 4) if multiplier else 0.0
    
    except (ValueError, IndexError):
        return 0.0


def build_text(title: str, description, features, details: dict) -> str:
    """
    Concatenate all text fields into one clean string.
    
    DECISION: We combine title + description + features + details
    because each field adds different information:
    - Title: product name and key attributes
    - Description: marketing copy and use cases  
    - Features: bullet points of specs
    - Details: structured attributes (dimensions, material, etc.)
    
    We remove part numbers because they cause memorization.
    We cap at 4000 chars to avoid extremely long inputs that
    slow down the LLM preprocessing step.
    """
    # Join list fields if they came as lists
    desc_text     = ' '.join(description) if isinstance(description, list) else str(description or '')
    features_text = ' '.join(features)    if isinstance(features, list)    else str(features or '')
    details_text  = json.dumps(details)   if isinstance(details, dict)     else str(details or '')
    
    # Combine all text
    combined = f"{title}\n{desc_text}\n{features_text}\n{details_text}"
    
    # Clean up whitespace
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    # Remove part numbers (they cause memorization not generalization)
    combined = re.sub(PART_NUMBER_PATTERN, '', combined)
    
    # Cap length to avoid runaway inputs
    return combined[:4000]


def clean_item(raw_item: dict, category: str) -> Optional[Product]:
    """
    Convert one raw HuggingFace datapoint to a clean Product.
    
    Returns None if the item should be excluded.
    Caller collects all non-None results.
    
    EXCLUSION REASONS:
    1. Price outside $0.50 - $999.49
    2. Price string can't be parsed (missing or malformed)
    3. Combined text too short (< 300 chars)
    """
    # --- Parse and validate price ---
    price = parse_price(raw_item.get('price'))
    
    if price is None:
        return None  # unparseable price
    
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None  # outside our target price range
    
    # --- Parse details JSON ---
    try:
        details = json.loads(raw_item.get('details', '{}') or '{}')
    except (json.JSONDecodeError, TypeError):
        details = {}
    
    # --- Build combined text ---
    text = build_text(
        title       = raw_item.get('title', ''),
        description = raw_item.get('description', []),
        features    = raw_item.get('features', []),
        details     = details
    )
    
    if len(text) < MIN_TEXT_LEN:
        return None  # too little text to learn from
    
    # --- Extract weight ---
    weight = extract_weight_in_pounds(details)
    
    return Product(
        title    = raw_item.get('title', ''),
        category = category,
        price    = price,
        text     = text,
        weight   = weight
    )


def clean_category(raw_dataset, category: str) -> list[Product]:
    """
    Clean all items in one category.
    
    Applies clean_item() to every raw datapoint.
    Filters out None results (rejected items).
    Reports how many items survived the filters.
    """
    products = []
    
    for raw_item in raw_dataset:
        product = clean_item(raw_item, category)
        if product is not None:
            products.append(product)
    
    # Report filtering stats
    total     = len(raw_dataset)
    kept      = len(products)
    rejected  = total - kept
    print(f"  {category}: {total:,} raw → {kept:,} clean "
          f"({rejected:,} rejected, {100*rejected/total:.1f}%)")
    
    return products
