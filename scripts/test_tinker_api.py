#!/usr/bin/env python3
"""
Test script for Tinker OpenAI-compatible API.

Tests both the completions and chat completions endpoints to determine
which one works with Tinker checkpoints.

Usage:
    uv run python scripts/test_tinker_api.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()

# Tinker API configuration
BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Known good checkpoint from the training run
MODEL_PATH = "tinker://488643ee-3be8-523e-9297-aecf5f8bb48f:train:0/sampler_weights/final"


def test_completions_api(client: OpenAI, model: str) -> bool:
    """Test the legacy completions API (client.completions.create)."""
    print("\n--- Testing completions API ---")
    try:
        response = client.completions.create(
            model=model,
            prompt="The capital of France is",
            max_tokens=50,
            temperature=0.7,
        )
        print(f"✓ Completions API works!")
        print(f"  Response: {response.choices[0].text}")
        return True
    except Exception as e:
        print(f"✗ Completions API failed: {e}")
        return False


def test_chat_completions_api(client: OpenAI, model: str) -> bool:
    """Test the chat completions API (client.chat.completions.create)."""
    print("\n--- Testing chat completions API ---")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        print(f"✓ Chat completions API works!")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Chat completions API failed: {e}")
        return False


def test_chat_completions_with_image(client: OpenAI, model: str) -> bool:
    """Test chat completions with an image (VLM format)."""
    import base64
    from io import BytesIO
    from PIL import Image

    print("\n--- Testing chat completions with image (VLM) ---")
    try:
        # Create a simple test image (1x1 red pixel)
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.7,
        )
        print(f"✓ Chat completions with image works!")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Chat completions with image failed: {e}")
        return False


def main():
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("ERROR: TINKER_API_KEY not set")
        sys.exit(1)

    print(f"Tinker API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_PATH}")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    # Test all APIs
    completions_ok = test_completions_api(client, MODEL_PATH)
    chat_ok = test_chat_completions_api(client, MODEL_PATH)
    vlm_ok = test_chat_completions_with_image(client, MODEL_PATH)

    print("\n--- Summary ---")
    print(f"Completions API: {'✓' if completions_ok else '✗'}")
    print(f"Chat Completions API: {'✓' if chat_ok else '✗'}")
    print(f"Chat + Image (VLM): {'✓' if vlm_ok else '✗'}")

    if not completions_ok and not chat_ok:
        print("\nBoth APIs failed. Check your API key and model path.")
        sys.exit(1)

    if vlm_ok:
        print("\n✓ Tinker checkpoint is ready for VLM agent evaluation!")
    elif chat_ok and not vlm_ok:
        print("\n⚠ WARNING: Tinker's OpenAI-compatible API does NOT support VLM (vision) messages.")
        print("  The API only accepts text-based messages, not image content.")
        print("  This means checkpoints cannot be evaluated via the OpenAI SDK for VLM agents.")
        print("\n  Options:")
        print("  1. Contact Tinker support about VLM inference support")
        print("  2. Use Tinker's native SDK if available for VLM inference")
        print("  3. Export the checkpoint weights for local inference")


if __name__ == "__main__":
    main()

