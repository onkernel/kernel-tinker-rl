#!/usr/bin/env python3
"""
Plot training metrics from Tinker RL runs.

Usage:
    uv run python -m scripts.plot_metrics results/              # all runs, comparison
    uv run python -m scripts.plot_metrics results/ -c           # all runs, continuation
    uv run python -m scripts.plot_metrics results/run_name/     # single run
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def discover_runs(path: Path) -> list[Path]:
    """Find metrics.jsonl files from a path."""
    if path.is_file() and path.name == "metrics.jsonl":
        return [path]
    if path.is_dir():
        direct = path / "metrics.jsonl"
        if direct.exists():
            return [direct]
        return sorted(path.glob("*/metrics.jsonl"), key=lambda p: p.parent.name)
    return []


def get_label(metrics_path: Path) -> str:
    """Extract a compact label from config or directory name."""
    config_path = metrics_path.parent / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            ds = cfg.get("dataset_builder", {})
            return f"bs{ds.get('batch_size', '?')}_gs{ds.get('group_size', '?')}_lr{cfg.get('learning_rate', '?')}"
        except Exception:
            pass
    return metrics_path.parent.name[-40:]


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("paths", nargs="*", default=[], help="Results directories or metrics.jsonl files")
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--metric", default="env/all/reward/total", help="Metric to plot")
    parser.add_argument("-c", "--continuation", action="store_true", help="Combine runs into one line")
    parser.add_argument("--title", help="Plot title")
    args = parser.parse_args()

    # Discover metrics files
    metrics_files: list[Path] = []
    for p in args.paths or [Path("/tmp/kernel-tinker-rl")]:
        path = Path(p)
        if path.exists():
            metrics_files.extend(discover_runs(path))

    if not metrics_files:
        print("No metrics.jsonl files found")
        return 1

    print(f"Found {len(metrics_files)} run(s)")

    # Load data
    runs: list[tuple[str, pd.DataFrame]] = []
    for mf in metrics_files:
        try:
            df = pd.read_json(mf, lines=True)
            if args.metric in df.columns:
                runs.append((get_label(mf), df))
        except Exception as e:
            print(f"Warning: {mf}: {e}")

    if not runs:
        print(f"No data found for metric: {args.metric}")
        return 1

    # Plot
    plt.figure(figsize=(12, 7))
    colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    if args.continuation and len(runs) > 1:
        steps, values, offset = [], [], 0
        for label, df in runs:
            steps.extend((df["step"] + offset).tolist())
            values.extend(df[args.metric].tolist())
            offset = steps[-1] + 1
        plt.plot(steps, values, "o-", markersize=3, lw=1.5, color=colors[0], label="Training")
        print(f"Plotted {len(steps)} total steps (continuation)")
    else:
        for i, (label, df) in enumerate(runs):
            plt.plot(df["step"], df[args.metric], "o-", markersize=3, lw=1.5, 
                     color=colors[i % 10], label=label, alpha=0.8)
            print(f"  {label}: {len(df)} steps")

    metric_name = args.metric.split("/")[-1].replace("_", " ").title()
    plt.xlabel("Training Step")
    plt.ylabel(metric_name)
    plt.title(args.title or f"RL Training: {metric_name}")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    if args.output:
        out = Path(args.output)
    elif len(metrics_files) == 1:
        out = metrics_files[0].parent / "reward_curve.png"
    else:
        out = Path("reward_curve.png" if args.continuation else "comparison_reward_curve.png")

    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
