"""

PURPOSE:
    Clean interface to get price predictions from the fine-tuned model.
    Two modes: local (needs GPU) or remote via Modal (no GPU needed).

USAGE:
    # Remote (recommended — no local GPU needed)
    from deployment.predict import RemotePredictor
    predictor = RemotePredictor()
    price = predictor.predict("Title: Sony WH-1000XM4...")
    
    # Local (needs GPU with 3GB+ VRAM)
    from deployment.predict import LocalPredictor
    predictor = LocalPredictor()
    predictor.load()
    price = predictor.predict("Title: Sony WH-1000XM4...")
"""

import re
import torch

# These must EXACTLY match what was used during fine-tuning
# Any change breaks predictions
QUESTION = "What does this cost to the nearest dollar?"
PRICE_PREFIX = "Price is $"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = "maulik78/pricer-2026-06-10_06.40.40-lite"


def format_prompt(description: str) -> str:
    """
    Format description as the inference prompt.
    """
    return f"{QUESTION}\n\n{description.strip()}\n\n{PRICE_PREFIX}"


def parse_price(generated_text: str) -> float:
    """
    Extract price from model's generated continuation.
    
    Model generates after "Price is $":
    Good: "278.00" or "278" or "278.00<extra tokens>"
    Bad:  "" (empty) or "N/A" or other garbage
    
    We use regex to find the first number pattern.
    Returns 0.0 if nothing parseable.
    """
    clean = generated_text.replace("$", "").replace(",", "").strip()
    match = re.search(r"\d+\.?\d*", clean)
    if match:
        price = float(match.group())
        if 0 < price < 10_000:  # sanity check
            return price
    return 0.0


class LocalPredictor:
    """
    Run predictions locally using the fine-tuned model.
    Requires GPU with ~3 GB VRAM.
    """

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """
        Load 4-bit quantized base model + LoRA adapters.
        Call this once before making predictions.
        Takes ~30 seconds on first call.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading model on: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Same quantization config as training
        quant_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_quant_type       = "nf4",
        )

        # Load frozen base model
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config = quant_config,
            device_map          = "auto",
        )

        # Attach LoRA adapter weights
        self.model = PeftModel.from_pretrained(base, FINETUNED_MODEL)
        self.model.eval()

        memory = self.model.get_memory_footprint() / 1e6
        print(f"Model loaded — {memory:.0f} MB")

    def predict(self, description: str) -> float:
        """
        Predict price for one product description.
        max_new_tokens=8: prices are short, cap to avoid extra tokens.
        do_sample=False: greedy decoding, most likely price.
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .predict()")

        prompt = format_prompt(description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = 8,
                    do_sample      = False,
                    pad_token_id   = self.tokenizer.eos_token_id,
                )

        # Keep only generated tokens (slice off the prompt)
        prompt_len     = inputs["input_ids"].shape[1]
        generated_ids  = output_ids[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return parse_price(generated_text)


class RemotePredictor:
    """
    Run predictions via Modal deployed service.
    No local GPU needed.
    
    First call: ~30-60 seconds (cold start)
    Subsequent calls: ~2-3 seconds (warm container)
    """

    def __init__(self):
        import modal
        Pricer      = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        print("Connected to Modal pricer service")

    def predict(self, description: str) -> float:
        """Send to Modal cloud, get price back."""
        result = self.pricer.price.remote(description)
        return float(result)


if __name__ == "__main__":
    # Quick test of the remote predictor
    predictor = RemotePredictor()

    test_items = [
        "Title: Sony WH-1000XM4 Wireless Headphones\nCategory: Electronics\nBrand: Sony\nDescription: Industry-leading noise cancelling headphones\nDetails: 30 hour battery, touch controls, multipoint",
        "Title: KitchenAid Stand Mixer 5Qt\nCategory: Appliances\nBrand: KitchenAid\nDescription: Professional stand mixer for home baking\nDetails: 10 speed settings, dough hook, flat beater",
        "Title: DeWalt 20V Cordless Drill\nCategory: Tools\nBrand: DeWalt\nDescription: Compact cordless drill driver kit\nDetails: Variable speed, LED light, 2 batteries included",
    ]

    print("Testing price predictions:\n")
    for item in test_items:
        price = predictor.predict(item)
        print(f"  {item[:60]}...")
        print(f"  Predicted: ${price:.2f}\n")
