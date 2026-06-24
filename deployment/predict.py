"""
INFERENCE: Run Predictions with the Fine-tuned Model
======================================================

PURPOSE:
    Simple, clean interface to run price predictions
    using the fine-tuned Llama 3.2-3B model.

TWO MODES:
    1. Local: Load model directly in this process
       → Use for development and testing
       → Requires GPU with 3GB+ VRAM
    
    2. Remote: Call the Modal deployed service
       → Use for production
       → No local GPU needed
       → ~2-3 second latency per prediction

THE INFERENCE PIPELINE:
    Text description
        ↓
    Format as training prompt
        ↓
    Tokenize (text → token IDs)
        ↓
    Generate (model predicts next tokens)
        ↓
    Decode (token IDs → text)
        ↓
    Parse price from generated text
        ↓
    Return float
"""

import re
import torch


# ── Prompt Format ─────────────────────────────────────────────
# CRITICAL: Must match the format used during fine-tuning exactly.
# The model learned to complete THIS specific prompt pattern.
# If we change the format, predictions will be garbage.
PROMPT_TEMPLATE = "What does this cost to the nearest dollar?\n\n{description}\n\nPrice is $"


def format_prompt(description: str) -> str:
    """
    Format a product description as a model input prompt.
    
    This must EXACTLY match the format used in training.
    During training, examples looked like:
    
    "What does this cost to the nearest dollar?
    
    Title: Sony WH-1000XM4 Wireless Headphones
    Category: Electronics
    Brand: Sony
    Description: Industry leading noise cancelling headphones
    Details: 30 hour battery, touch sensor controls
    
    Price is $278.00"
    
    At inference, we give everything up to and including
    "Price is $" and the model completes with the price.
    """
    return PROMPT_TEMPLATE.format(description=description.strip())


def parse_generated_price(generated_text: str) -> float:
    """
    Extract price from model's generated continuation.
    
    The model generates text after "Price is $".
    We expect something like "278.00" or "278" or "$278.00".
    
    We use regex to find the first number because the model
    sometimes generates extra tokens after the price.
    
    Returns 0.0 if no valid price found.
    """
    # Remove any $ signs the model might regenerate
    cleaned = generated_text.replace('$', '').replace(',', '').strip()
    
    # Find first number (int or decimal)
    match = re.search(r'\d+\.?\d*', cleaned)
    
    if match:
        price = float(match.group())
        # Sanity check: prices should be in our training range
        if 0 < price < 10_000:
            return price
    
    return 0.0


class LocalPredictor:
    """
    Run predictions locally using the fine-tuned model.
    
    Loads the 4-bit quantized base model + LoRA adapter weights.
    Requires GPU with ~3GB VRAM.
    
    Usage:
        predictor = LocalPredictor()
        predictor.load()
        price = predictor.predict("Title: Sony headphones...")
    """
    
    def __init__(
        self,
        base_model:      str = "meta-llama/Llama-3.2-3B",
        finetuned_model: str = "maulik78/pricer-2026-06-10_06.40.40-lite"
    ):
        self.base_model      = base_model
        self.finetuned_model = finetuned_model
        self.model           = None
        self.tokenizer       = None
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        """
        Load tokenizer, quantized base model, and LoRA adapters.
        
        Loading order matters:
        1. Load tokenizer (small, fast)
        2. Load base model in 4-bit (2.2GB)
        3. Attach LoRA adapters on top (73MB)
        
        The base model weights are FROZEN — only LoRA adapters
        are active during inference, just as during training.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        print(f"Loading on: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 4-bit quantization config
        # NF4 (NormalFloat4) is better than standard int4 for neural
        # network weights because they follow a normal distribution
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,    # quantize the quant constants too
            bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bfloat16
            bnb_4bit_quant_type="nf4"          # NormalFloat4 format
        )
        
        # Load frozen base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        # Attach fine-tuned LoRA adapters
        self.model = PeftModel.from_pretrained(base, self.finetuned_model)
        self.model.eval()  # disable dropout, enable inference mode
        
        memory = self.model.get_memory_footprint() / 1e6
        print(f"✅ Model loaded — Memory: {memory:.0f} MB")
    
    def predict(self, description: str) -> float:
        """
        Predict price for one product description.
        
        max_new_tokens=8: A price like "$1,299.99" is at most
        6-7 tokens. We cap at 8 to avoid generating extra text
        and to save computation.
        
        do_sample=False: Deterministic greedy decoding.
        We want the most likely price, not creative variety.
        """
        if self.model is None:
            raise RuntimeError("Call load() before predict()")
        
        prompt = format_prompt(description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8,    # prices are short
                    do_sample=False,     # deterministic
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Slice off the prompt tokens — keep only generated tokens
        prompt_length  = inputs["input_ids"].shape[1]
        generated_ids  = output_ids[0, prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return parse_generated_price(generated_text)


class RemotePredictor:
    """
    Run predictions via the Modal deployed service.
    
    No local GPU needed. Sends the description to Modal's
    cloud GPU and receives the predicted price.
    
    First call: ~30-60 seconds (cold start, model loading)
    Subsequent calls: ~2-3 seconds (warm container)
    
    Usage:
        predictor = RemotePredictor()
        price = predictor.predict("Title: Sony headphones...")
    """
    
    def __init__(self):
        import modal
        Pricer      = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        print("✅ Connected to Modal pricer service")
    
    def predict(self, description: str) -> float:
        """Send description to Modal cloud, get price back."""
        result = self.pricer.price.remote(description)
        return float(result)
