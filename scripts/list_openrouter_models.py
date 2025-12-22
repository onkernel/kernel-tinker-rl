#!/usr/bin/env python3
"""
List all available OpenRouter models.

This script fetches and displays all models available through the OpenRouter API,
with optional filtering and sorting capabilities.

Usage:
    # List all models
    uv run python -m scripts.list_openrouter_models

    # Filter by name (case-insensitive)
    uv run python -m scripts.list_openrouter_models --filter qwen

    # Show only free models
    uv run python -m scripts.list_openrouter_models --free

    # Sort by price (cheapest first)
    uv run python -m scripts.list_openrouter_models --sort price

    # Sort by context length (largest first)
    uv run python -m scripts.list_openrouter_models --sort context --desc

    # Show detailed info for a specific model
    uv run python -m scripts.list_openrouter_models --filter gpt-4o --verbose

Environment Variables:
    OPENROUTER_API_KEY: Optional, but may be required for some API features
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


def fetch_models(api_key: str | None = None) -> list[dict]:
    """Fetch all available models from OpenRouter API."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(OPENROUTER_API_URL, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except httpx.HTTPError as e:
        console.print(f"[red]Error fetching models: {e}[/]")
        sys.exit(1)


def format_price(price: float | str | None) -> str:
    """Format price per million tokens."""
    if price is None:
        return "[green]Free[/]"
    # Convert to float if string
    try:
        price_float = float(price)
    except (ValueError, TypeError):
        return "?"
    if price_float == 0:
        return "[green]Free[/]"
    # Price is per token, convert to per million
    price_per_million = price_float * 1_000_000
    if price_per_million < 0.01:
        return f"${price_per_million:.4f}/M"
    return f"${price_per_million:.2f}/M"


def format_context(context_length: int | None) -> str:
    """Format context length in K tokens."""
    if context_length is None:
        return "?"
    if context_length >= 1_000_000:
        return f"{context_length // 1_000_000}M"
    return f"{context_length // 1000}K"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="List available OpenRouter models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--filter", "-f",
        default=None,
        help="Filter models by name (case-insensitive)",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Show only free models",
    )
    parser.add_argument(
        "--sort", "-s",
        choices=["name", "price", "context"],
        default="name",
        help="Sort models by field (default: name)",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed model information",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of results",
    )

    args = parser.parse_args()

    # Fetch models
    api_key = os.getenv("OPENROUTER_API_KEY")
    console.print("[dim]Fetching models from OpenRouter...[/]")
    models = fetch_models(api_key)

    if not models:
        console.print("[yellow]No models found[/]")
        return 1

    # Filter models
    if args.filter:
        filter_lower = args.filter.lower()
        models = [m for m in models if filter_lower in m.get("id", "").lower()]

    def is_free(model: dict) -> bool:
        """Check if a model is free."""
        pricing = model.get("pricing", {})
        try:
            prompt = float(pricing.get("prompt", 0) or 0)
            completion = float(pricing.get("completion", 0) or 0)
            return prompt == 0 and completion == 0
        except (ValueError, TypeError):
            return False

    if args.free:
        models = [m for m in models if is_free(m)]

    def safe_float(val: str | float | None, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Sort models
    if args.sort == "name":
        models.sort(key=lambda m: m.get("id", ""), reverse=args.desc)
    elif args.sort == "price":
        models.sort(
            key=lambda m: (
                safe_float(m.get("pricing", {}).get("prompt")),
                safe_float(m.get("pricing", {}).get("completion")),
            ),
            reverse=args.desc,
        )
    elif args.sort == "context":
        models.sort(
            key=lambda m: m.get("context_length", 0) or 0,
            reverse=args.desc,
        )

    # Limit results
    if args.limit:
        models = models[:args.limit]

    # Display results
    if args.verbose:
        for model in models:
            console.print(f"\n[bold cyan]{model.get('id', 'Unknown')}[/]")
            console.print(f"  Name: {model.get('name', 'N/A')}")
            console.print(f"  Context: {format_context(model.get('context_length'))}")

            pricing = model.get("pricing", {})
            prompt_price = pricing.get("prompt", 0) or 0
            completion_price = pricing.get("completion", 0) or 0
            console.print(f"  Pricing:")
            console.print(f"    Input:  {format_price(prompt_price)}")
            console.print(f"    Output: {format_price(completion_price)}")

            if model.get("description"):
                desc = model["description"][:200]
                if len(model["description"]) > 200:
                    desc += "..."
                console.print(f"  Description: {desc}")

            if model.get("top_provider"):
                console.print(f"  Top Provider: {model['top_provider']}")
    else:
        table = Table(title=f"OpenRouter Models ({len(models)} total)")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Context", justify="right")
        table.add_column("Input Price", justify="right")
        table.add_column("Output Price", justify="right")

        for model in models:
            pricing = model.get("pricing", {})
            prompt_price = pricing.get("prompt", 0) or 0
            completion_price = pricing.get("completion", 0) or 0

            table.add_row(
                model.get("id", "Unknown"),
                format_context(model.get("context_length")),
                format_price(prompt_price),
                format_price(completion_price),
            )

        console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

