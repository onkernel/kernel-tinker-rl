#!/usr/bin/env python3
"""
Merge LoRA adapter weights into a base model using PEFT.

This script takes a PEFT LoRA adapter (e.g., from Tinker training) and merges
it into the base model, creating a single model with the fine-tuned weights
baked in. This is necessary because vLLM doesn't support all LoRA target
modules (like MoE expert layers).

Usage:
    # Merge locally (requires ~80GB GPU memory for 30B model)
    uv run python -m scripts.merge_lora \
        --base-model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --lora-path ./checkpoints/final \
        --output-path ./merged_model

    # Or run on Modal (recommended for large models)
    uv run modal run scripts/modal_vllm_serve.py::merge_lora_weights
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--lora-path",
        default="./checkpoints/final",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path",
        default="./merged_model",
        help="Output directory for the merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading (auto, cpu, cuda, etc.)",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> None:
    """
    Merge LoRA adapter weights into the base model.

    Args:
        base_model_path: HuggingFace model ID or local path
        lora_path: Path to the LoRA adapter directory
        output_path: Output directory for merged model
        dtype: Data type for model weights
        device_map: Device mapping strategy
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = get_torch_dtype(dtype)

    print(f"\n{'='*60}")
    print("LoRA Merge Configuration")
    print(f"{'='*60}")
    print(f"Base model: {base_model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Output: {output_path}")
    print(f"Dtype: {dtype}")
    print(f"Device map: {device_map}")
    print(f"{'='*60}\n")

    # Step 1: Load the base model
    # Use AutoModelForVision2Seq for VLMs like Qwen3-VL
    print("Step 1/4: Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"  ✓ Base model loaded ({base_model.num_parameters() / 1e9:.1f}B params)")

    # Step 2: Load tokenizer/processor
    print("\nStep 2/4: Loading tokenizer/processor...")
    try:
        # Try loading processor (for VLMs)
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        print("  ✓ Processor loaded")
    except Exception:
        processor = None
        print("  ℹ No processor found, loading tokenizer only")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    print("  ✓ Tokenizer loaded")

    # Step 3: Load and merge LoRA adapter
    print("\nStep 3/4: Loading and merging LoRA adapter...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch_dtype,
    )
    print("  ✓ LoRA adapter loaded")

    # Merge weights
    merged_model = model_with_lora.merge_and_unload()
    print("  ✓ LoRA weights merged into base model")

    # Step 4: Save merged model
    print(f"\nStep 4/4: Saving merged model to {output_path}...")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    print("  ✓ Model weights saved")

    tokenizer.save_pretrained(output_dir)
    print("  ✓ Tokenizer saved")

    if processor is not None:
        processor.save_pretrained(output_dir)
        print("  ✓ Processor saved")

    # Copy over any additional config files needed for VLMs
    base_path = Path(base_model_path)
    config_files = [
        "preprocessor_config.json",
        "image_processor_config.json",
        "chat_template.json",
        "generation_config.json",
    ]
    for config_file in config_files:
        src = base_path / config_file
        dst = output_dir / config_file
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"  ✓ Copied {config_file}")

    # Also copy any .py files (for custom model code with trust_remote_code)
    for py_file in base_path.glob("*.py"):
        dst = output_dir / py_file.name
        if not dst.exists():
            shutil.copy(py_file, dst)
            print(f"  ✓ Copied {py_file.name}")

    print("\n" + "="*60)
    print("✓ Merge complete!")
    print(f"  Merged model saved to: {output_dir}")
    print(f"  Size: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e9:.1f} GB")
    print("="*60 + "\n")


def main() -> int:
    args = parse_args()

    # Validate LoRA path
    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        print(f"Error: LoRA path does not exist: {lora_path}")
        return 1

    adapter_config = lora_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json found in {lora_path}")
        return 1

    # Run merge
    try:
        merge_lora(
            base_model_path=args.base_model,
            lora_path=str(lora_path),
            output_path=args.output_path,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        return 0
    except Exception as e:
        print(f"\nError during merge: {e}")
        raise


if __name__ == "__main__":
    import sys
    sys.exit(main())

