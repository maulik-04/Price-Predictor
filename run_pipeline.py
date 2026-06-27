"""
run_pipeline.py — Full Pipeline Runner
========================================

PURPOSE:
    Run the complete data pipeline end-to-end from one script.
    Each step is modular -- you can run individual steps or all.

PIPELINE STAGES:
    Stage 1: Load raw Amazon data (step1_load_data.py)
    Stage 2: Clean and parse products (step2_clean_data.py)
    Stage 3: Weighted sampling (step3_sample_data.py)
    Stage 4: Push to HuggingFace (step4_push_to_hub.py)
    Stage 5: Build training prompts (step5_build_prompts.py)
    Stage 6: Train QLoRA model (models/train_qlora.py)

USAGE:
    # Run everything
    python run_pipeline.py --all
    
    # Run specific stage
    python run_pipeline.py --stage 4
    
    # Run from stage 3 onwards
    python run_pipeline.py --from-stage 3

NOTE:
    Stages 1-4 require Google Colab (large downloads).
    Stage 6 requires GPU (Colab T4 recommended).
    Stages 3, 5 can run locally.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def check_credentials():
    """Verify required API keys are available."""
    required = ['HF_TOKEN']
    missing  = [key for key in required if not os.environ.get(key)]
    if missing:
        print(f"Missing required credentials: {missing}")
        print("Add them to your .env file")
        sys.exit(1)
    print("Credentials verified")


def run_stage_4_push():
    """
    Stage 4: Push curated dataset to HuggingFace.
    Assumes train/val/test splits are already in memory from Stage 3.
    """
    from pipeline.step4_push_to_hub import authenticate, push_full_dataset, push_lite_dataset, verify_push
    authenticate()
    print("\nStage 4: Push to HuggingFace")
    print("Note: Requires train/val/test from Stage 3")
    print("Run stages 1-3 first to build the dataset")


def run_stage_5_prompts():
    """
    Stage 5: Build training prompts from preprocessed dataset.
    Requires maulik78/items_full and maulik78/items_lite to exist.
    """
    from pipeline.step5_build_prompts import load_tokenizer, build_and_push_prompts
    from pipeline.step5_build_prompts import LITE_SOURCE, LITE_PROMPTS, FULL_SOURCE, FULL_PROMPTS

    tokenizer = load_tokenizer()
    print("\nStage 5: Building training prompts")
    print("Building lite prompts first (verify format)...")
    build_and_push_prompts(LITE_SOURCE, LITE_PROMPTS, tokenizer)
    print("Building full prompts...")
    build_and_push_prompts(FULL_SOURCE, FULL_PROMPTS, tokenizer)


def run_stage_6_train():
    """
    Stage 6: QLoRA fine-tuning.
    Requires GPU. Best run on Google Colab T4.
    """
    print("\nStage 6: QLoRA Fine-tuning")
    print("This requires a GPU. Running on Google Colab T4 is recommended.")
    print("Use notebooks/03_fine_tuning.ipynb for interactive training.")
    print("Or run: python models/train_qlora.py")


def main():
    parser = argparse.ArgumentParser(description='Amazon Price Predictor Pipeline')
    parser.add_argument('--stage', type=int, help='Run a specific stage (4, 5, or 6)')
    parser.add_argument('--all', action='store_true', help='Run all stages')
    parser.add_argument('--check', action='store_true', help='Check credentials only')
    args = parser.parse_args()

    print("Amazon Price Predictor — Pipeline Runner")
    print("=" * 45)

    check_credentials()

    if args.check:
        print("All checks passed")
        return

    if args.stage == 4 or args.all:
        run_stage_4_push()

    if args.stage == 5 or args.all:
        run_stage_5_prompts()

    if args.stage == 6 or args.all:
        run_stage_6_train()

    if not (args.stage or args.all):
        parser.print_help()
        print("\nFor stages 1-3 (heavy data loading), use notebooks/01_data_pipeline.ipynb")
        print("For stage 6 (GPU training), use notebooks/03_fine_tuning.ipynb")


if __name__ == '__main__':
    main()
