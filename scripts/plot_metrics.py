#!/usr/bin/env python3
"""
plot_metrics.py - Plot training metrics from a Tinker RL run.

Usage:
    uv run python -m scripts.plot_metrics /tmp/kernel-tinker-rl/<run_name>/metrics.jsonl
    uv run python -m scripts.plot_metrics  # Uses most recent run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_latest_metrics() -> Path | None:
    """Find the most recent metrics.jsonl file."""
    base_dir = Path("/tmp/kernel-tinker-rl")
    if not base_dir.exists():
        return None
    
    # Find all metrics.jsonl files and sort by modification time
    metrics_files = list(base_dir.glob("*/metrics.jsonl"))
    if not metrics_files:
        return None
    
    return max(metrics_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default=None,
        help="Path to metrics.jsonl (default: most recent run)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: same directory as metrics)",
    )
    args = parser.parse_args()

    # Find metrics file
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
    else:
        metrics_path = find_latest_metrics()
        if metrics_path is None:
            print("No metrics.jsonl found in /tmp/kernel-tinker-rl/")
            sys.exit(1)
        print(f"Using most recent: {metrics_path}")

    if not metrics_path.exists():
        print(f"File not found: {metrics_path}")
        sys.exit(1)

    # Load metrics
    df = pd.read_json(metrics_path, lines=True)
    print(f"Loaded {len(df)} training steps")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = metrics_path.parent / "reward_curve.png"

    # Plot reward curve
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["env/all/reward/total"], marker="o", markersize=4, label="reward/total")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("RL Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to PNG
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()


