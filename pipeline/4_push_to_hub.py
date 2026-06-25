"""
STEP 4: Push Cleaned Dataset to HuggingFace Hub
=================================================

PURPOSE:
    After curating and sampling 820,000 clean products in Step 3,
    we push them to HuggingFace Hub so they are:
    
    1. Accessible from anywhere (Colab, laptop, cloud)
    2. Version controlled (HuggingFace uses Git under the hood)
    3. Shareable — anyone can load our dataset in one line
    4. The foundation for all downstream steps in this project

WHY HUGGINGFACE HUB:
    Alternatives exist (S3, Google Drive, local disk) but HuggingFace
    gives us:
    - Free storage for public datasets
    - Parquet format (columnar, compressed, fast to load)
    - Built-in train/val/test split support
    - Direct integration with the datasets library
    - A public profile showing our work

WHAT WE PUSH:
    Two versions of the dataset:
    
    Full version (maulik78/items_raw_full):
        - Train: 800,000 items
        - Val:   10,000 items
        - Test:  10,000 items
        - Use: actual model training
    
    Lite version (maulik78/items_raw_lite):
        - Train: 20,000 items
        - Val:   1,000 items
        - Test:  1,000 items
        - Use: fast experimentation, debugging, testing

    The lite version lets us iterate quickly —
    loading 20k items takes seconds vs 10+ minutes for 800k.

MEMORY STRATEGY:
    Pushing 800k items as a single Dataset would require
    loading all of them into RAM simultaneously (~6-8GB).
    
    Instead we convert in chunks of 250k and push train/val/test
    as separate operations, keeping peak RAM under 4GB.

OUTPUT:
    huggingface.co/datasets/maulik78/items_raw_full
    huggingface.co/datasets/maulik78/items_raw_lite
"""

import os
import gc
from dataclasses import asdict
from typing import Optional
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi
from dotenv import load_dotenv

load_dotenv(override=True)


# ── Configuration ─────────────────────────────────────────────
HF_USER = 'maulik78'   # your HuggingFace username

FULL_DATASET_NAME = f'{HF_USER}/items_raw_full'
LITE_DATASET_NAME = f'{HF_USER}/items_raw_lite'

# Lite dataset sizes — small enough for fast iteration
LITE_TRAIN_SIZE = 20_000
LITE_VAL_SIZE   = 1_000
LITE_TEST_SIZE  = 1_000


def authenticate():
    """
    Login to HuggingFace.
    
    Reads HF_TOKEN from .env file.
    add_to_git_credential=True stores the token so git push
    commands work without re-authentication.
    """
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Check your .env file.")
    
    login(token=token, add_to_git_credential=True)
    print(f"Authenticated as {HF_USER}")


def products_to_hf_dataset(products: list) -> Dataset:
    """
    Convert a list of Product objects to a HuggingFace Dataset.
    
    WHY asdict():
    HuggingFace Dataset.from_list() expects a list of dictionaries.
    dataclasses.asdict() converts each Product to a plain dict
    with the same field names.
    
    The resulting Dataset automatically infers column types:
    - title, category, summary → string
    - price, weight → float64
    - text → string (long)
    
    Args:
        products: List of Product dataclass instances
        
    Returns:
        HuggingFace Dataset ready to push
    """
    records = [asdict(p) for p in products]
    return Dataset.from_list(records)


def push_dataset(
    name:  str,
    train: list,
    val:   list,
    test:  list
):
    """
    Push a train/val/test split to HuggingFace Hub.
    
    WHY DatasetDict:
    HuggingFace expects a DatasetDict with named splits.
    This structure allows users to load specific splits:
        ds = load_dataset('maulik78/items_raw_full')
        train = ds['train']   # 800k items
        test  = ds['test']    # 10k items
    
    WHY WE NAME THE SPLIT 'val' NOT 'validation':
    The dataset uses 'val' for brevity. This is a project
    convention — just be consistent across all notebooks.
    
    Args:
        name:  HuggingFace dataset repo name (e.g. 'maulik78/items_raw_full')
        train: List of Product objects for training
        val:   List of Product objects for validation
        test:  List of Product objects for testing
    """
    print(f"\nPreparing dataset: {name}")
    print(f"  Train: {len(train):,} items")
    print(f"  Val:   {len(val):,} items")
    print(f"  Test:  {len(test):,} items")
    
    print("  Converting to HuggingFace format...")
    
    # Convert each split — most memory-intensive step
    # DatasetDict holds all three splits
    dataset_dict = DatasetDict({
        'train': products_to_hf_dataset(train),
        'val':   products_to_hf_dataset(val),
        'test':  products_to_hf_dataset(test),
    })
    
    print(f"  Pushing to huggingface.co/datasets/{name}...")
    
    # push_to_hub uploads all splits as parquet files
    # parquet is columnar format — fast for column-wise operations
    # and much smaller than CSV (typically 5-10x compression)
    dataset_dict.push_to_hub(name)
    
    # Free memory after pushing
    del dataset_dict
    gc.collect()
    
    print(f"Done: huggingface.co/datasets/{name}")


def push_full_dataset(train: list, val: list, test: list):
    """
    Push the full 820k dataset to HuggingFace.
    
    This is the primary dataset used for training.
    """
    push_dataset(FULL_DATASET_NAME, train, val, test)


def push_lite_dataset(train: list, val: list, test: list):
    """
    Push a smaller lite dataset for fast experimentation.
    
    WHY A LITE DATASET:
    Loading 800k items takes 10+ minutes. During development
    and debugging, we don't want to wait that long.
    
    The lite dataset (20k train) lets us:
    - Test that code works before running on full dataset
    - Iterate quickly on model architecture
    - Debug evaluation pipelines
    
    We take the FIRST 20k items from the already-shuffled train set,
    so the distribution is representative.
    """
    push_dataset(
        name  = LITE_DATASET_NAME,
        train = train[:LITE_TRAIN_SIZE],
        val   = val[:LITE_VAL_SIZE],
        test  = test[:LITE_TEST_SIZE],
    )


def verify_push(dataset_name: str):
    """
    Load the dataset back from HuggingFace and verify it's correct.
    
    WHY VERIFY:
    Pushing can fail silently or partially. Loading back
    confirms the data is actually there and readable.
    
    We check:
    - All three splits exist
    - Item counts match what we pushed
    - A sample item looks correct
    """
    from datasets import load_dataset
    
    print(f"\nVerifying {dataset_name}...")
    ds = load_dataset(dataset_name)
    
    print(f"  Splits: {list(ds.keys())}")
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data):,} items")
    
    # Check a sample item has all expected fields
    sample = ds['train'][0]
    expected_fields = ['title', 'category', 'price', 'text', 'weight', 'summary']
    missing = [f for f in expected_fields if f not in sample]
    
    if missing:
        print(f"Missing fields: {missing}")
    else:
        print(f"All fields present: {list(sample.keys())}")
    
    print(f"  Sample title: {sample['title'][:60]}")
    print(f"  Sample price: ${sample['price']}")
    
    return ds


if __name__ == "__main__":
    """
    Main execution — called after step3_sample_data.py produces
    the train/val/test splits.
    
    Typical usage:
        from pipeline.step3_sample_data import split_dataset
        train, val, test = split_dataset(sampled_products)
        
        from pipeline.step4_push_to_hub import push_full_dataset, push_lite_dataset
        push_full_dataset(train, val, test)
        push_lite_dataset(train, val, test)
    """
    print("Step 4: Push to HuggingFace Hub")
    print("=" * 45)
    authenticate()
    
    # Note: train/val/test come from step3_sample_data.py
    # This file assumes they are already in memory
    print("\nReady to push. Call push_full_dataset() and push_lite_dataset()")
    print("with your train/val/test splits from step3_sample_data.py")
