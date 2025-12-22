#!/usr/bin/env python3
"""
Modal SGLang server for serving fine-tuned Qwen3-VL models.

SGLang is recommended by Qwen for serving Qwen3 models and has better
support for MoE architectures than vLLM.

Usage:
    # Step 1: Upload LoRA adapter to Modal volume
    uv run modal run scripts/modal_sglang_serve.py::upload_lora_adapter \
        --local-path ./checkpoints/final

    # Step 2: Merge LoRA into base model (runs on Modal GPU)
    uv run modal run scripts/modal_sglang_serve.py::merge_lora_weights

    # Step 3: Deploy the server
    uv run modal deploy scripts/modal_sglang_serve.py

    # The server will be available at:
    # https://kernel--qwen3-vl-sglang-serve.modal.run/v1/chat/completions

    # Test with run_agent.py:
    uv run python -m scripts.run_agent \
        --env agent_auth --random \
        --base-url https://kernel--qwen3-vl-sglang-serve.modal.run/v1

Environment Variables:
    MODAL_TOKEN_ID: Required for Modal deployment
    MODAL_TOKEN_SECRET: Required for Modal deployment
    HF_TOKEN: Optional, for downloading gated models
"""

import subprocess

import modal

# Modal app configuration
APP_NAME = "qwen3-vl-sglang"

# Model configuration
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MERGED_MODEL_NAME = "merged-finetuned"  # Name for merged model in volume
GPU_TYPE = "H100"
GPU_COUNT = 1

# Volumes for caching
MODEL_CACHE_DIR = "/model-cache"
LORA_DIR = "/lora-adapters"

# Create Modal app
app = modal.App(APP_NAME)

# Create volumes for persistent storage
model_volume = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("sglang-lora-adapters", create_if_missing=True)

# Build image with SGLang using the official Docker image
# This includes all CUDA dependencies and SGLang properly configured
# Note: We don't add_python because the Docker image already has Python + SGLang installed
sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.6.post2-cu129-amd64",
    )
    .run_commands(
        "pip install 'huggingface_hub[hf_transfer]>=0.36.0,<1.0' 'hf_transfer>=0.1.9'",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": MODEL_CACHE_DIR,
    })
)

# Build image with PEFT for merging (includes transformers, peft, torch)
merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.9.1",
        "torchvision>=0.24.1",  # Required for VLM processor
        "transformers>=4.57.3",
        "peft>=0.18.0",
        "accelerate>=1.12.0",
        "safetensors>=0.7.0",
        "huggingface_hub[hf_transfer]>=0.36.0,<1.0",
        "hf_transfer>=0.1.9",
        "qwen-vl-utils>=0.0.14",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": MODEL_CACHE_DIR,
    })
)


@app.function(
    image=sglang_image,
    volumes={MODEL_CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_dotenv()],
    timeout=3600,
)
def prefetch_model():
    """Pre-download the base model to cache."""
    import os

    from huggingface_hub import snapshot_download

    print(f"Downloading base model: {BASE_MODEL}")
    snapshot_download(
        BASE_MODEL,
        local_dir=f"{MODEL_CACHE_DIR}/{BASE_MODEL.replace('/', '--')}",
        token=os.getenv("HF_TOKEN"),
    )
    model_volume.commit()
    print("✓ Model download complete!")


@app.local_entrypoint()
def upload_lora_adapter(local_path: str = "./checkpoints/final", adapter_name: str = "finetuned"):
    """Upload a LoRA adapter from local machine to the Modal volume."""
    from pathlib import Path

    local_lora = Path(local_path)
    if not local_lora.exists():
        raise ValueError(f"LoRA path does not exist: {local_path}")

    # Read all files and their contents
    files_data: dict[str, bytes] = {}
    for f in local_lora.iterdir():
        if f.is_file():
            files_data[f.name] = f.read_bytes()
            print(f"  Read: {f.name} ({len(files_data[f.name]) / 1e6:.1f} MB)")

    print(f"\nUploading {len(files_data)} files to Modal...")
    # Call the remote upload function with file data
    _upload_lora_files.remote(files_data, adapter_name)
    print("✓ Upload complete!")


@app.function(
    image=sglang_image,
    volumes={LORA_DIR: lora_volume},
    timeout=1800,  # 30 min for large uploads
)
def _upload_lora_files(files_data: dict[str, bytes], adapter_name: str = "finetuned"):
    """Internal function to write files to volume."""
    from pathlib import Path

    remote_path = Path(LORA_DIR) / adapter_name
    remote_path.mkdir(parents=True, exist_ok=True)

    # Write each file
    for filename, content in files_data.items():
        file_path = remote_path / filename
        file_path.write_bytes(content)
        print(f"  Wrote: {filename} ({len(content) / 1e6:.1f} MB)")

    lora_volume.commit()
    print(f"✓ LoRA adapter saved to {remote_path}")


@app.function(
    image=merge_image,
    gpu=GPU_TYPE,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        LORA_DIR: lora_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,  # 2 hours for large model operations
)
def merge_lora_weights(
    adapter_name: str = "finetuned",
    output_name: str = MERGED_MODEL_NAME,
):
    """
    Merge LoRA adapter weights into the base model.

    This creates a new model with the LoRA weights baked in, which can then
    be served directly by SGLang without needing runtime LoRA support.

    Args:
        adapter_name: Name of the LoRA adapter in the lora volume
        output_name: Name for the merged model in the model volume
    """
    import os
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

    base_model_path = f"{MODEL_CACHE_DIR}/{BASE_MODEL.replace('/', '--')}"
    lora_path = Path(LORA_DIR) / adapter_name
    output_path = Path(MODEL_CACHE_DIR) / output_name

    print("\n" + "=" * 60)
    print("LoRA Merge Configuration")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Base model path: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output path: {output_path}")
    print("=" * 60 + "\n")

    # Validate LoRA adapter exists
    if not lora_path.exists():
        raise ValueError(
            f"LoRA adapter not found at {lora_path}. "
            "Run upload_lora_adapter first."
        )

    adapter_config = lora_path / "adapter_config.json"
    if not adapter_config.exists():
        raise ValueError(f"No adapter_config.json found in {lora_path}")

    # Download base model if not present
    if not Path(base_model_path).exists():
        print("Step 0/4: Downloading base model...")
        snapshot_download(
            BASE_MODEL,
            local_dir=base_model_path,
            token=os.getenv("HF_TOKEN"),
        )
        print("  ✓ Base model downloaded")

    # Step 1: Load the base model
    # Use AutoModelForVision2Seq for VLMs like Qwen3-VL
    print("\nStep 1/4: Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  ✓ Base model loaded ({base_model.num_parameters() / 1e9:.1f}B params)")

    # Step 2: Load tokenizer/processor
    print("\nStep 2/4: Loading tokenizer/processor...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    print("  ✓ Processor loaded")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    print("  ✓ Tokenizer loaded")

    # Step 3: Load and merge LoRA adapter
    print("\nStep 3/4: Loading and merging LoRA adapter...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        str(lora_path),
        torch_dtype=torch.bfloat16,
    )
    print("  ✓ LoRA adapter loaded")

    # Merge weights
    merged_model = model_with_lora.merge_and_unload()
    print("  ✓ LoRA weights merged into base model")

    # Step 4: Save merged model
    print(f"\nStep 4/4: Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
    )
    print("  ✓ Model weights saved")

    tokenizer.save_pretrained(output_path)
    print("  ✓ Tokenizer saved")

    processor.save_pretrained(output_path)
    print("  ✓ Processor saved")

    # Copy any missing config files from base model (for VLM support)
    import shutil
    config_files = [
        "preprocessor_config.json",
        "chat_template.json",
        "generation_config.json",
    ]
    base_path = Path(base_model_path)
    for config_file in config_files:
        src = base_path / config_file
        dst = output_path / config_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {config_file} from base model")

    # Commit changes to volume
    model_volume.commit()

    # Calculate size
    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )

    print("\n" + "=" * 60)
    print("✓ Merge complete!")
    print(f"  Merged model saved to: {output_path}")
    print(f"  Size: {total_size / 1e9:.1f} GB")
    print("=" * 60 + "\n")


@app.function(
    image=sglang_image,
    gpu=GPU_TYPE,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        LORA_DIR: lora_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
    scaledown_window=300,  # 5 min idle timeout
    timeout=3600,
)
@modal.web_server(port=30000, startup_timeout=600)
def serve():
    """
    Run SGLang's OpenAI-compatible server.

    Will use the merged model if available, otherwise falls back to base model.

    Endpoints:
    - POST /v1/chat/completions - Chat completions
    - GET /health - Health check
    """
    from pathlib import Path

    # Check for merged model first
    merged_model_path = Path(MODEL_CACHE_DIR) / MERGED_MODEL_NAME
    base_model_path = Path(MODEL_CACHE_DIR) / BASE_MODEL.replace("/", "--")

    if merged_model_path.exists() and any(merged_model_path.glob("*.safetensors")):
        model_path = str(merged_model_path)
        print(f"✓ Using merged model: {model_path}")
    elif base_model_path.exists():
        model_path = str(base_model_path)
        print(f"ℹ Using base model (no merged model found): {model_path}")
        print("  To use fine-tuned weights, run: modal run scripts/modal_sglang_serve.py::merge_lora_weights")
    else:
        # Fall back to downloading from HuggingFace
        model_path = BASE_MODEL
        print(f"ℹ Using HuggingFace model (will download): {model_path}")

    # Launch SGLang server
    # SGLang uses python -m sglang.launch_server for the OpenAI-compatible API
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--port", "30000",
        "--trust-remote-code",
        # Memory and performance settings
        "--mem-fraction-static", "0.85",
        # Logging
        "--log-level", "info",
    ]

    print(f"Starting SGLang server: {' '.join(cmd)}")
    subprocess.Popen(cmd)

