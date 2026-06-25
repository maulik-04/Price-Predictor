"""
STEP 1: Load Raw Amazon Product Data
=====================================

PURPOSE:
    Download raw Amazon product listings from HuggingFace.
    This is the entry point of our entire pipeline.

WHY THIS DATASET:
    McAuley-Lab/Amazon-Reviews-2023 from UC San Diego contains
    48 million products across 33 categories. It's one of the few
    publicly available large-scale product datasets with both
    rich text descriptions AND actual prices.

WHY THESE 8 CATEGORIES:
    Selected for diversity + quality:
    - Electronics, Musical_Instruments: technical specs correlate
      strongly with price (CPU speed, materials etc.)
    - Automotive: largest category — will be down-weighted later
    - Toys_and_Games, Appliances: wide price range ($5 - $999)
    - Office_Products, Tools: clear functional descriptions
    - Cell_Phones_and_Accessories: strong brand price signals

OUTPUT:
    Raw dataset objects ready for Step 2 (cleaning)
"""

import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(override=True)


# ── Authentication ─────────────────────────────────────────────
# HuggingFace login needed to access Llama (used in fine-tuning)
# and to push our cleaned datasets back to our account
login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)


# ── Category Selection ─────────────────────────────────────────
# We chose categories where text descriptions contain strong
# price signals. A detailed description of a "professional
# grade stainless steel commercial espresso machine" tells you
# a lot more about price than "coffee maker".
CATEGORIES = [
    'Automotive',                  # 974k items — largest, diverse price range
    'Electronics',                 # 464k items — rich technical specs
    'Office_Products',             # 249k items — consistent descriptions
    'Tools_and_Home_Improvement',  # 552k items — specific measurements
    'Cell_Phones_and_Accessories', # 242k items — brand = strong price signal
    'Toys_and_Games',              # 348k items — wide $5-$500 range
    'Appliances',                  # 35k items  — smallest, high quality
    'Musical_Instruments',         # 69k items  — brand + materials matter
]


def load_category(category_name: str):
    """
    Load a single Amazon product category from HuggingFace.
    
    The dataset is stored as parquet files on HuggingFace Hub.
    Each category downloads as a separate file (~0.3GB to 5GB).
    
    We use split='full' because we want ALL items — we'll do
    our own train/val/test split after cleaning and sampling.
    
    Args:
        category_name: One of the 8 category strings above
        
    Returns:
        HuggingFace Dataset object with raw product fields:
        - title, description, features, details, price, etc.
    """
    print(f"Loading {category_name}...")
    
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",  # dataset repo on HuggingFace
        f"raw_meta_{category_name}",         # config name = "raw_meta_" + category
        split="full",                        # load all items, not just train/test
        trust_remote_code=True               # needed for custom loading script
    )
    
    print(f"  ✅ {category_name}: {len(dataset):,} raw items")
    return dataset


def inspect_raw_item(dataset, index=0):
    """
    Print a raw item to understand the data structure.
    
    IMPORTANT: Raw items are messy.
    - 'price' is a string like '$49.99' — needs parsing
    - 'description' is a list of strings — needs joining
    - 'details' is a JSON string — needs parsing
    - Part numbers and product codes pollute the text
    
    This is why Step 2 (cleaning) is necessary.
    """
    item = dataset[index]
    print("\nRaw item structure:")
    for key, value in item.items():
        value_str = str(value)[:100]
        print(f"  {key:<15}: {value_str}")


if __name__ == "__main__":
    # Load just one category to inspect the structure
    sample = load_category("Appliances")
    inspect_raw_item(sample)
    print(f"\nTotal raw items in Appliances: {len(sample):,}")
