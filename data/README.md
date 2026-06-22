# Datasets

All datasets for this project are hosted on HuggingFace.
No data files are stored in this repository due to size.

## Dataset Versions

### Raw Datasets (after curation)
- **Full:** [maulik78/items_raw_full](https://huggingface.co/datasets/maulik78/items_raw_full) — 820k items
- **Lite:** [maulik78/items_raw_lite](https://huggingface.co/datasets/maulik78/items_raw_lite) — 22k items

Each item contains:
- `title` — product title
- `category` — one of 8 Amazon categories
- `price` — actual price ($0.50–$999.49)
- `full` — concatenated raw product text
- `weight` — product weight in pounds

### Preprocessed Datasets (after LLM cleaning)
- **Full:** [maulik78/items_full](https://huggingface.co/datasets/maulik78/items_full) — 820k items
- **Lite:** [maulik78/items_lite](https://huggingface.co/datasets/maulik78/items_lite) — 22k items

Each item contains:
- `title`, `category`, `price`, `weight` — same as raw
- `summary` — LLM-generated clean structured description
- `full` — set to None (no longer needed)

### Prompt Datasets (for fine-tuning)
- **Full:** [maulik78/items_prompts_full](https://huggingface.co/datasets/maulik78/items_prompts_full) — 820k items
- **Lite:** [maulik78/items_prompts_lite](https://huggingface.co/datasets/maulik78/items_prompts_lite) — 22k items

Each item contains:
- `prompt` — full training prompt including product description
- `completion` — target price string (e.g., "64.00")

## Loading Datasets

```python
from datasets import load_dataset

# Load preprocessed dataset
dataset = load_dataset("maulik78/items_full")
train = dataset['train']    # 800,000 items
val   = dataset['val']      # 10,000 items
test  = dataset['test']     # 10,000 items

# Load prompt dataset for fine-tuning
dataset = load_dataset("maulik78/items_prompts_lite")
print(dataset['train'][0])
# {'prompt': 'What does this cost to the nearest dollar?\n\nTitle: ...', 
#  'completion': '64.00'}
```

## Categories

The 8 Amazon product categories used:

| Category | Raw Items | After Sampling |
|----------|-----------|----------------|
| Automotive | 974,469 | ~48,000 (down-weighted 95%) |
| Electronics | 464,024 | ~193,000 |
| Tools and Home Improvement | 552,147 | ~188,000 |
| Toys and Games | 347,657 | ~107,000 |
| Office Products | 248,767 | ~72,000 |
| Musical Instruments | 68,855 | ~37,000 |
| Cell Phones and Accessories | 242,351 | ~36,000 |
| Appliances | 35,307 | ~15,000 |
| **Total** | **2,933,577** | **820,000** |
