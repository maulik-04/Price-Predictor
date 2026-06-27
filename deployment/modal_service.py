"""
Modal Deployment: Llama 3.2-3B as a Serverless Cloud API
=========================================================

PURPOSE:
    Deploy the fine-tuned model as a persistent REST API
    that anyone can call without a local GPU.

WHY MODAL (not AWS/GCP/local):
    Modal provides serverless GPU compute.
    You write Python, Modal handles:
    - Docker containers with all dependencies
    - GPU allocation on demand
    - Scaling (more containers under load)
    - Cold start management
    - Zero cost when idle

ARCHITECTURE:
    Your laptop/Colab                Modal cloud
    ───────────────         ──────────────────────────────
    pricer.price.remote(text)  -->  Container starts (cold start ~30s)
                                    Load 4-bit base model
                                    Load LoRA adapter weights
                                    Run inference
                               <--  Return predicted price

COLD START vs WARM:
    Cold start: container spun up from scratch -> ~30-60 seconds
    Warm:       container already running -> ~2-3 seconds
    scaledown_window=120: keep container alive 2 min after last call

CLASS-BASED vs FUNCTION-BASED:
    Function-based: model MAY reload on every call (bad)
    Class-based (@app.cls): model loads ONCE in @modal.enter(),
    then reused for every .price() call (good)

SECRETS:
    We don't hardcode API keys in code.
    Modal reads HF_TOKEN from a secret named 'huggingface'
    that you create at: modal.com/secrets

DEPLOYMENT:
    modal deploy deployment/modal_service.py
    
    After first deployment:
    Pricer = modal.Cls.from_name("pricer-service", "Pricer")
    pricer = Pricer()
    price  = pricer.price.remote("Sony WH-1000XM4 headphones...")
"""

import modal

# Modal App definition
# All functions/classes in this file belong to this app
app = modal.App("pricer-service")

# Docker image: what software to install in the container
# Each .pip_install() call adds packages to the container
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "transformers>=4.47.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.3.0",
        "torch>=2.5.0",
        "huggingface_hub>=0.27.0",
    ])
)

# Model identifiers
BASE_MODEL      = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = "maulik78/pricer-2026-06-10_06.40.40-lite"

# Prompt format — must exactly match training format
QUESTION     = "What does this cost to the nearest dollar?"
PRICE_PREFIX = "Price is $"


@app.cls(
    image   = image,
    gpu     = "T4",           # use T4 GPU in Modal's cloud
    secrets = [modal.Secret.from_name("huggingface")],
    timeout = 600,            # max 10 min per request
    scaledown_window = 120,   # keep alive 2 min after last call
)
class Pricer:
    """
    Serverless price prediction class.
    
    @modal.enter() runs ONCE when the container starts.
    This is where we load the model — expensive but only done once.
    All subsequent .price() calls reuse the loaded model.
    
    Without @modal.enter(), the model would reload on every call
    which would take 30+ seconds per prediction.
    """

    @modal.enter()
    def setup(self):
        """
        Called once when container starts. Load model here.
        
        Loading order:
        1. Login to HuggingFace (needed for gated Llama model)
        2. Load tokenizer (fast)
        3. Load base model in 4-bit quantization (slow, ~30s)
        4. Load LoRA adapter weights on top (fast, ~5s)
        5. Set to eval mode (disables dropout for inference)
        """
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login

        # Authenticate -- HF_TOKEN comes from Modal secret
        hf_token = os.environ.get("HF_TOKEN")
        login(token=hf_token)
        print("Logged in to HuggingFace")

        # Check GPU
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # 4-bit quantization (same config as training)
        quant_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_quant_type       = 'nf4',
        )

        # Load frozen base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config = quant_config,
            device_map          = 'auto',
            token               = hf_token,
        )
        base_model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        # Load LoRA adapter weights on top of frozen base
        self.model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, token=hf_token)
        self.model.eval()  # disable dropout for inference

        memory = self.model.get_memory_footprint() / 1e6
        print(f"Setup complete. Memory: {memory:.0f} MB")
        self.device = "cuda"

    @modal.method()
    def price(self, description: str) -> float:
        """
        Predict price for one product description.
        
        INFERENCE PIPELINE:
        1. Format as prompt (same format as training)
        2. Tokenize (text -> token IDs)
        3. Generate (model predicts next tokens)
        4. Slice off prompt tokens (keep only generated tokens)
        5. Decode (token IDs -> text)
        6. Parse price from generated text
        
        max_new_tokens=8:
            A price like "$1,299.99" is 5-6 tokens max.
            Capping at 8 avoids generating extra text.
        
        do_sample=False:
            Greedy decoding -- always take the most likely token.
            We want the most probable price, not creative variety.
        
        Args:
            description: Product summary text
            
        Returns:
            Predicted price as float
        """
        import torch
        import re

        # Build prompt (must match training format exactly)
        prompt = f"{QUESTION}\n\n{description.strip()}\n\n{PRICE_PREFIX}"

        # Tokenize and move to GPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate price tokens
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = 8,
                    do_sample      = False,
                    pad_token_id   = self.tokenizer.eos_token_id,
                )

        # Slice off prompt tokens -- keep only what the model generated
        prompt_len     = inputs["input_ids"].shape[1]
        generated_ids  = output_ids[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse price from generated text
        # The model generates something like "278.00" or "278" or "278.00<|eot_id|>"
        clean = generated_text.replace("$", "").replace(",", "").strip()
        match = re.search(r"\d+\.?\d*", clean)
        price = float(match.group()) if match else 0.0

        print(f"Input: {description[:50]}... -> ${price:.2f}")
        return price
