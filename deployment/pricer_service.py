
import modal

app = modal.App("pricer-service")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "transformers>=4.47.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.3.0",
        "torch>=2.5.0",
        "huggingface_hub>=0.27.0",
        "scipy",
    ])
)

BASE_MODEL      = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = "maulik78/pricer-2026-06-10_06.40.40-lite"

@app.cls(
    image=image,
    gpu="T4",
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=600,
    scaledown_window=120,
)
class Pricer:

    @modal.enter()
    def setup(self):
        import os
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )
        from peft import PeftModel
        from huggingface_hub import login

        # Login with token from secret
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print(f"HuggingFace login successful")
        else:
            print("HF_TOKEN not found in environment")
            raise ValueError("HF_TOKEN secret not found")

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            token=hf_token
        )
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print("Tokenizer loaded")

        print("Loading base model in 4-bit...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token
        )
        base_model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        print("Base model loaded")

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            FINETUNED_MODEL,
            token=hf_token
        )
        self.model.eval()
        print("LoRA adapter loaded")

        self.device = "cuda"
        memory = self.model.get_memory_footprint()
        print(f"Setup complete — Memory: {memory/1e6:.0f} MB")

    @modal.method()
    def price(self, description: str) -> float:
        import torch
        import re

        prompt = f"What does this cost to the nearest dollar?\n\n{description}\n\nPrice is $"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        prompt_len = inputs["input_ids"].shape[1]
        generated  = output_ids[0, prompt_len:]
        result     = self.tokenizer.decode(generated, skip_special_tokens=True)

        result = result.replace("$","").replace(",","").strip()
        match  = re.search(r"[\d.]+", result)
        price  = float(match.group()) if match else 0.0

        print(f"Input: {description[:50]}... → ${price:.2f}")
        return price
