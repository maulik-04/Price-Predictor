"""

PURPOSE:
    Fine-tune Meta's Llama 3.2-3B on Amazon pricing data
    using QLoRA on a free Google Colab T4 GPU.

THE PROBLEM WITH STANDARD FINE-TUNING:
    Full fine-tuning of Llama 3.2-3B needs ~28 GB VRAM.
    Free T4 has 15 GB. Impossible without QLoRA.

QLORA = Quantization + LoRA

    Quantization: Store weights in 4-bit NF4 format.
        6.4 GB -> 2.2 GB (3x memory reduction)
        Computations still done in bfloat16 for stability.

    LoRA (Low-Rank Adaptation): Only train small adapter matrices.
        Instead of: W_new = W_old + delta_W  (delta_W = 3072x3072 = huge)
        LoRA does:  W_new = W_old + alpha * (A x B)
        Where A = (3072 x 32), B = (32 x 3072), r=32 is the rank.
        3B trainable params -> 18M trainable params (166x reduction)

    Together: entire fine-tuning fits in 3-4 GB VRAM.

WHY WE TARGET ATTENTION LAYERS:
    Attention learns WHICH words to focus on.
    Fine-tuning attention reshapes focus toward pricing signals:
    brand names, quality tier, category, specific features.

TRAINING DECISIONS:
    epochs=1:          One pass through 20k items prevents overfitting
    batch_size=4:      Limited by T4 VRAM (15 GB)
    grad_accum=8:      Effective batch = 4x8 = 32
    lr=1e-4:           Standard for LoRA fine-tuning
    cosine schedule:   LR decays smoothly from 1e-4 to ~0
    max_grad_norm=0.3: Gradient clipping for stability

"""

import os
import torch
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_MODEL = 'meta-llama/Llama-3.2-3B'
HF_USER = 'maulik78'
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}-lite"
PROJECT_RUN_NAME = f"pricer-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"
DATASET_NAME = f'{HF_USER}/items_prompts_lite'
VAL_SIZE = 500

# LoRA hyperparameters
LORA_R = 32
LORA_ALPHA = 64      # scale = alpha/r = 2.0
LORA_DROPOUT = 0.1
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Training hyperparameters
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8       # effective batch = 4x8 = 32
MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
WEIGHT_DECAY = 0.001
LR_SCHEDULER = 'cosine'
OPTIMIZER = 'paged_adamw_32bit'
MAX_GRAD_NORM = 0.3
LOG_STEPS = 5
SAVE_STEPS = 100


def check_gpu():
    """
    Check GPU and determine compute dtype.
    bfloat16: available on A100+ (capability >= 8), more stable
    float16:  T4 (capability 7.5), slightly less stable but works
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Enable GPU in Colab runtime settings.")
    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")
    return use_bf16


def load_dataset_splits():
    
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME)
    train = ds['train']
    val = ds['val'].select(range(VAL_SIZE))
    print(f"Train: {len(train):,} | Val: {len(val):,}")
    print(f"\nSample:\n{train[0]['prompt'][:200]}")
    return train, val


def build_quant_config(use_bf16: bool):
    
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type = 'nf4',
    )


def load_model_and_tokenizer(quant_config):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = quant_config,
        device_map = 'auto',
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"Memory: {model.get_memory_footprint()/1e6:.0f} MB")
    return model, tokenizer


def build_lora_config():
    
    from peft import LoraConfig
    config = LoraConfig(
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        target_modules = TARGET_MODULES,
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, scale={LORA_ALPHA/LORA_R:.1f}x")
    return config


def build_training_config(use_bf16: bool):
    
    from trl import SFTConfig
    return SFTConfig(
        output_dir = PROJECT_RUN_NAME,
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        optim = OPTIMIZER,
        save_steps = SAVE_STEPS,
        save_total_limit = 3,
        logging_steps = LOG_STEPS,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        fp16 = not use_bf16,
        bf16 = use_bf16,
        max_grad_norm = MAX_GRAD_NORM,
        warmup_ratio = WARMUP_RATIO,
        lr_scheduler_type = LR_SCHEDULER,
        report_to = 'wandb',
        run_name = RUN_NAME,
        max_length = MAX_SEQUENCE_LENGTH,
        save_strategy = 'steps',
        hub_strategy = 'every_save',
        push_to_hub = True,
        hub_model_id = HUB_MODEL_NAME,
        hub_private_repo = False,
        eval_strategy = 'steps',
        eval_steps = SAVE_STEPS,
        dataset_text_field = 'prompt',
    )


def show_trainable_params(model):
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%) <- LoRA adapters")
    print(f"Frozen:    {total-trainable:,} ({100*(total-trainable)/total:.2f}%) <- base model")


def run_training(model, tokenizer, lora_config, train_config, train_data, val_data):
    """
    Execute the training loop.

    Per-batch training cycle:
    1. Forward: model predicts next tokens from prompt
    2. Loss: cross-entropy on PRICE tokens only (not prompt tokens)
    3. Backward: gradients flow only to LoRA adapters (frozen base = no grad)
    4. Accumulate for GRADIENT_ACCUMULATION steps
    5. Clip gradients (max_norm=0.3)
    6. Step: update LoRA weights with paged AdamW
    7. Repeat

    Monitoring:
        Training loss decreasing: model is learning
        Val loss < train loss: good generalization
        Val loss >> train loss: overfitting, consider stopping
    """
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model         = model,
        train_dataset = train_data,
        eval_dataset  = val_data,
        peft_config   = lora_config,
        args          = train_config,
    )
    show_trainable_params(trainer.model)
    print(f"\nTraining -> huggingface.co/{HUB_MODEL_NAME}\n")
    trainer.train()
    print("\nTraining complete!")
    return trainer


if __name__ == '__main__':
    from huggingface_hub import login
    import wandb

    login(os.environ['HF_TOKEN'], add_to_git_credential=True)
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        wandb.login()
        os.environ['WANDB_PROJECT'] = 'price-predictor'

    use_bf16             = check_gpu()
    train_data, val_data = load_dataset_splits()
    quant_config         = build_quant_config(use_bf16)
    lora_config          = build_lora_config()
    train_config         = build_training_config(use_bf16)
    model, tokenizer     = load_model_and_tokenizer(quant_config)

    trainer = run_training(model, tokenizer, lora_config, train_config, train_data, val_data)
    trainer.model.push_to_hub(HUB_MODEL_NAME, private=False)

    if wandb_key:
        wandb.finish()

    print(f"\nModel: huggingface.co/{HUB_MODEL_NAME}")
    print(f"FINETUNED_MODEL = '{HUB_MODEL_NAME}'")
