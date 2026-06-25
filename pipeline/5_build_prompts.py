"""
STEP 5: Build Training Prompts for LLM Fine-tuning
====================================================

PURPOSE:
    Convert cleaned product records into prompt-completion pairs
    that teach Llama 3.2-3B to predict prices.

THE CORE IDEA:
    Fine-tuning works by showing the model many examples of:
        INPUT:  "What does this cost? [product description]"
        OUTPUT: "Price is $278.00"
    
    After seeing 20,000 such examples, the model learns to
    complete the prompt with a realistic price.

WHY THIS SPECIFIC PROMPT FORMAT:
    "What does this cost to the nearest dollar?\n\n{description}\n\nPrice is $"
    
    Three reasons:
    1. CONSISTENCY — the model must see the EXACT same format
       at inference time as during training. Even small changes
       (extra space, different punctuation) degrade performance.
    
    2. NATURAL LANGUAGE — framing it as a question makes it
       feel like instruction-following, which Llama was
       trained on during its instruction tuning phase.
    
    3. COMPLETION STYLE — ending with "Price is $" prompts the
       model to generate just the number, not a long explanation.

THE DIFFERENCE BETWEEN TRAIN AND TEST PROMPTS:
    
    Training prompt (includes the answer — model learns from this):
        "What does this cost to the nearest dollar?
        
        Title: Sony WH-1000XM4 Headphones
        Category: Electronics
        ...
        
        Price is $278.00"
    
    Test prompt (no answer — model must predict this):
        "What does this cost to the nearest dollar?
        
        Title: Sony WH-1000XM4 Headphones
        Category: Electronics
        ...
        
        Price is $"
        ↑ model continues from here

WHY WE TOKENIZE DURING PROMPT BUILDING:
    Different tokenizers split text differently.
    We use Llama's tokenizer to count tokens so we can:
    - Identify summaries that are too long
    - Truncate them to fit within MAX_SEQUENCE_LENGTH
    - Ensure no training example exceeds the context window

TOKEN LENGTH DECISIONS:
    MAX_SEQUENCE_LENGTH = 128 tokens
    
    WHY 128 (not 256 or 512):
    Most cleaned summaries are 60-100 tokens.
    The question + prefix adds ~20 tokens.
    Total: ~80-120 tokens per example.
    Setting limit to 128 truncates <5% of examples
    while keeping training fast (longer sequences = slower training).

OUTPUT:
    Two datasets pushed to HuggingFace:
    - maulik78/items_prompts_full — 820k prompt-completion pairs
    - maulik78/items_prompts_lite — 22k prompt-completion pairs
    
    Each example has:
    - 'prompt': the full training prompt (question + description + answer)
    - 'completion': just the price number (for evaluation)
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(override=True)


# ── Configuration ──────────────────────────────────────────────
HF_USER = 'maulik78'

# Source datasets (created in Step 4)
FULL_SOURCE = f'{HF_USER}/items_full'   # preprocessed with LLM summaries
LITE_SOURCE = f'{HF_USER}/items_lite'

# Destination datasets
FULL_PROMPTS = f'{HF_USER}/items_prompts_full'
LITE_PROMPTS = f'{HF_USER}/items_prompts_lite'

# The base model we will fine-tune
# We load its tokenizer to count tokens accurately
BASE_MODEL = 'meta-llama/Llama-3.2-3B'

# Maximum tokens per training example
# Affects: training speed, GPU memory, how much text fits
MAX_SEQUENCE_LENGTH = 128

# Prefix that ends the prompt — model generates after this
PRICE_PREFIX = 'Price is $'

# The question that starts every prompt
QUESTION = 'What does this cost to the nearest dollar?'


# ── Prompt Templates ───────────────────────────────────────────

def build_training_prompt(summary: str, price: float) -> str:
    """
    Build a complete training example (question + description + answer).
    
    This is shown to the model during fine-tuning.
    The model learns to predict the price from the description.
    
    FORMAT:
        What does this cost to the nearest dollar?
        
        [product summary]
        
        Price is $[price]
    
    WHY round(price):
        We round to nearest dollar during training.
        This makes the target simpler — the model only needs
        to predict integers, not exact decimals like $49.73.
        In practice this costs very little accuracy since
        product prices tend to be round numbers anyway.
    
    Args:
        summary: LLM-cleaned product description
        price:   Actual product price in dollars
        
    Returns:
        Complete training prompt as a string
    """
    return (
        f"{QUESTION}\n\n"
        f"{summary.strip()}\n\n"
        f"{PRICE_PREFIX}{round(price)}.00"
    )


def build_test_prompt(summary: str) -> str:
    """
    Build an inference prompt (question + description, NO answer).
    
    This is used at test time — we give the model the question
    and description, then let it generate the price.
    
    The prompt ends with 'Price is $' so the model knows
    to generate a number next.
    
    CRITICAL: This must match the END of the training prompt
    exactly (up to but not including the price).
    If we change this at test time, predictions will be wrong.
    
    Args:
        summary: LLM-cleaned product description
        
    Returns:
        Prompt string ending with 'Price is $'
    """
    return (
        f"{QUESTION}\n\n"
        f"{summary.strip()}\n\n"
        f"{PRICE_PREFIX}"
    )


# ── Tokenizer for Length Control ───────────────────────────────

def load_tokenizer():
    """
    Load Llama's tokenizer for accurate token counting.
    
    WHY WE NEED THE ACTUAL TOKENIZER:
    Different tokenizers split text differently.
    "noise-cancelling" might be:
        - GPT tokenizer:   ['noise', '-', 'cancelling'] = 3 tokens
        - Llama tokenizer: ['noise', '-cancel', 'ling'] = 3 tokens
    
    Only Llama's tokenizer tells us how many tokens Llama will see.
    
    TOKENIZER SETTINGS:
    pad_token = eos_token:
        Llama has no dedicated padding token.
        We reuse the end-of-sequence token for padding.
        This is the standard workaround for Llama models.
    
    padding_side = 'right':
        Pad sequences on the right side.
        Required by the SFTTrainer used in Notebook 3.
    """
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    print(f"✅ Tokenizer loaded")
    print(f"   Vocabulary size: {tokenizer.vocab_size:,} tokens")
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """
    Count how many tokens Llama's tokenizer produces for a text.
    
    Used to check if a prompt will fit within MAX_SEQUENCE_LENGTH.
    We check BEFORE training to avoid silent truncation during
    the training loop.
    
    Args:
        text:      Any string
        tokenizer: Llama tokenizer instance
        
    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def truncate_summary(summary: str, tokenizer, max_tokens: int) -> str:
    """
    Truncate a summary to fit within the token limit.
    
    WHY TRUNCATE SUMMARIES NOT PROMPTS:
    The summary is the variable-length part. The question
    and price prefix are fixed. So we truncate the summary
    when needed to keep the total prompt length manageable.
    
    WHY TRUNCATE BY TOKENS NOT CHARACTERS:
    Different parts of text tokenize differently.
    "supercalifragilistic" might be 5-8 tokens.
    "the cat sat" might be 4 tokens.
    Character limits are imprecise for token budget control.
    
    APPROACH:
    We tokenize the summary, keep only max_tokens tokens,
    then decode back to text. This is exact.
    
    Args:
        summary:    Product description to potentially truncate
        tokenizer:  Llama tokenizer
        max_tokens: Maximum tokens to keep
        
    Returns:
        Potentially truncated summary string
    """
    tokens    = tokenizer.encode(summary, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return summary  # no truncation needed
    
    # Keep only the first max_tokens tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text   = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


# ── Main Prompt Building ────────────────────────────────────────

def build_prompts_for_split(
    items:          list,
    tokenizer,
    include_answer: bool,
    split_name:     str
) -> list:
    """
    Convert a list of product records to prompt-completion pairs.
    
    WHY include_answer CONTROLS TRAIN VS TEST:
    During training:  include_answer=True  → model sees full example
    During testing:   include_answer=False → model must predict
    
    This is the fundamental difference between supervised training
    and evaluation. Training examples teach. Test examples assess.
    
    WHAT WE STORE:
    For each item we store:
    - 'prompt': the full text fed to the model
      (includes answer for train, excludes for test)
    - 'completion': just the price number as a string
      (used during evaluation to compare predicted vs actual)
    
    TOKEN BUDGET:
    Question prefix   ≈ 20 tokens  (fixed overhead)
    Summary           ≈ 60-100 tokens (variable)
    Price prefix+num  ≈ 5-8 tokens  (fixed overhead)
    Total budget      = MAX_SEQUENCE_LENGTH = 128 tokens
    
    So we allow summary to use MAX_SEQUENCE_LENGTH - 30 tokens.
    
    Args:
        items:          List of product dicts from HuggingFace Dataset
        tokenizer:      Llama tokenizer for length control
        include_answer: True for train/val, False for test
        split_name:     'train', 'val', or 'test' (for logging)
        
    Returns:
        List of dicts with 'prompt' and 'completion' keys
    """
    from tqdm import tqdm
    
    results           = []
    truncated_count   = 0
    
    # Token budget for the summary
    # Total budget - overhead for question + price prefix
    SUMMARY_TOKEN_BUDGET = MAX_SEQUENCE_LENGTH - 30
    
    print(f"Building {split_name} prompts for {len(items):,} items...")
    
    for item in tqdm(items, desc=f'{split_name} prompts'):
        
        # Get the cleaned summary (or fall back to raw text)
        summary = item.get('summary') or item.get('text', '')
        price   = item['price']
        
        # Check and truncate summary if needed
        summary_tokens = count_tokens(summary, tokenizer)
        if summary_tokens > SUMMARY_TOKEN_BUDGET:
            summary = truncate_summary(summary, tokenizer, SUMMARY_TOKEN_BUDGET)
            truncated_count += 1
        
        # Build the appropriate prompt type
        if include_answer:
            # Training prompt: full example with price
            # Model learns to predict price from this
            prompt = build_training_prompt(summary, price)
        else:
            # Test prompt: stops before the price
            # Model must generate the price
            prompt = build_test_prompt(summary)
        
        # Store both the prompt and the target price
        # completion is used during evaluation to compare predictions
        results.append({
            'prompt':     prompt,
            'completion': f"{round(price)}.00",
        })
    
    pct_truncated = 100 * truncated_count / len(items) if items else 0
    print(f"  ✅ {len(results):,} prompts built")
    print(f"  ⚡ {truncated_count:,} summaries truncated ({pct_truncated:.1f}%)")
    
    # If more than 10% truncated, consider raising MAX_SEQUENCE_LENGTH
    if pct_truncated > 10:
        print(f"  ⚠️  High truncation rate. Consider increasing MAX_SEQUENCE_LENGTH")
    
    return results


def build_and_push_prompts(source_dataset: str, target_dataset: str, tokenizer):
    """
    Load a preprocessed dataset, build prompts, push to HuggingFace.
    
    SPLIT HANDLING:
    - train: include_answer=True  (model learns from these)
    - val:   include_answer=True  (used to monitor training loss)
    - test:  include_answer=False (model predicts, we evaluate)
    
    WHY val HAS ANSWERS:
    During training, we periodically compute validation loss.
    This requires the model to see the full example (with answer)
    so we can measure how well it predicts the price token.
    
    At EVALUATION TIME (Notebook 4), we use the test set
    with include_answer=False to get real predictions.
    
    Args:
        source_dataset: HuggingFace dataset with 'summary' and 'price' fields
        target_dataset: Where to push the prompt dataset
        tokenizer:      Llama tokenizer
    """
    print(f"\nBuilding prompts from: {source_dataset}")
    
    # Load the preprocessed dataset
    ds = load_dataset(source_dataset)
    
    # Build prompts for each split
    # Note: train and val get answers, test does not
    train_prompts = build_prompts_for_split(
        items          = list(ds['train']),
        tokenizer      = tokenizer,
        include_answer = True,   # model sees the price during training
        split_name     = 'train'
    )
    
    val_prompts = build_prompts_for_split(
        items          = list(ds['val']),
        tokenizer      = tokenizer,
        include_answer = True,   # validation loss needs the target
        split_name     = 'val'
    )
    
    test_prompts = build_prompts_for_split(
        items          = list(ds['test']),
        tokenizer      = tokenizer,
        include_answer = False,  # test: model must predict the price
        split_name     = 'test'
    )
    
    # Push to HuggingFace
    print(f"\nPushing to {target_dataset}...")
    DatasetDict({
        'train': Dataset.from_list(train_prompts),
        'val':   Dataset.from_list(val_prompts),
        'test':  Dataset.from_list(test_prompts),
    }).push_to_hub(target_dataset)
    
    print(f"✅ Pushed: huggingface.co/datasets/{target_dataset}")
    
    # Show a sample to verify format is correct
    print(f"\nSample training prompt:")
    print("─" * 50)
    print(train_prompts[0]['prompt'])
    print("─" * 50)
    print(f"Completion: {train_prompts[0]['completion']}")
    
    print(f"\nSample test prompt (no answer):")
    print("─" * 50)
    print(test_prompts[0]['prompt'])
    print("─" * 50)
    print("(model generates here)")


def analyze_token_lengths(source_dataset: str, tokenizer):
    """
    Analyze token length distribution before building prompts.
    
    WHY RUN THIS FIRST:
    Understanding the distribution helps us choose the right
    MAX_SEQUENCE_LENGTH. If 95% of examples are under 100 tokens
    but we set limit to 256, we waste GPU memory on padding.
    If 50% are over our limit, we're losing too much information.
    
    Good target: truncate < 5% of examples.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"Analyzing token lengths in {source_dataset}...")
    
    ds    = load_dataset(source_dataset, split='train')
    
    # Sample 5000 items for analysis (full dataset would be slow)
    sample_size = min(5000, len(ds))
    sample      = list(ds.select(range(sample_size)))
    
    token_counts = []
    for item in sample:
        summary = item.get('summary') or item.get('text', '')
        n_tokens = count_tokens(summary, tokenizer)
        token_counts.append(n_tokens)
    
    token_counts = np.array(token_counts)
    
    print(f"\nToken length statistics (sample of {sample_size:,} items):")
    print(f"  Mean:   {token_counts.mean():.1f} tokens")
    print(f"  Median: {np.median(token_counts):.1f} tokens")
    print(f"  95th %: {np.percentile(token_counts, 95):.1f} tokens")
    print(f"  Max:    {token_counts.max():.1f} tokens")
    
    # Show how different cutoffs affect truncation
    print(f"\nImpact of MAX_SEQUENCE_LENGTH on truncation rate:")
    for cutoff in [64, 96, 128, 160, 192, 256]:
        pct_truncated = 100 * (token_counts > cutoff - 30).mean()
        print(f"  Cutoff={cutoff:>4}: {pct_truncated:.1f}% truncated", 
              "← our choice" if cutoff == MAX_SEQUENCE_LENGTH else "")
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    plt.hist(token_counts, bins=50, color='steelblue', edgecolor='white', rwidth=0.8)
    plt.axvline(x=MAX_SEQUENCE_LENGTH-30, color='red', linestyle='--',
                label=f'Summary budget ({MAX_SEQUENCE_LENGTH-30} tokens)')
    plt.title('Token Length Distribution of Product Summaries', fontsize=13)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Main execution for Step 5.
    
    Run this after Step 4 has pushed the preprocessed datasets.
    
    Flow:
        1. Load Llama tokenizer
        2. Analyze token lengths (optional but informative)
        3. Build and push lite prompts (fast, verify format)
        4. Build and push full prompts (slow, ~30 mins for 800k)
    """
    print("Step 5: Build Training Prompts")
    print("=" * 45)
    
    # Authenticate
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    login(token=token, add_to_git_credential=True)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Analyze lengths first (optional)
    print("\n[Optional] Analyzing token lengths to validate MAX_SEQUENCE_LENGTH...")
    analyze_token_lengths(LITE_SOURCE, tokenizer)
    
    # Build and push lite prompts first (verify format is correct)
    print("\n[1/2] Building lite prompts...")
    build_and_push_prompts(LITE_SOURCE, LITE_PROMPTS, tokenizer)
    
    # Build and push full prompts
    print("\n[2/2] Building full prompts (~30 minutes for 800k items)...")
    build_and_push_prompts(FULL_SOURCE, FULL_PROMPTS, tokenizer)
    
    print("\n" + "="*50)
    print("✅ STEP 5 COMPLETE — PROMPTS BUILT AND PUSHED")
    print("="*50)
    print(f"\nDatasets available:")
    print(f"  huggingface.co/datasets/{LITE_PROMPTS}")
    print(f"  huggingface.co/datasets/{FULL_PROMPTS}")
    print(f"\nNext: Use these in Notebook 3 for QLoRA fine-tuning")
