#!/usr/bin/env python3
"""
Download checkpoint weights from Tinker.

This script downloads LoRA adapter weights from a Tinker training run
for local hosting or deployment to Modal.

Usage:
    # Download the final checkpoint from a training run
    uv run python -m scripts.download_checkpoint \
        --checkpoint "tinker://488643ee-3be8-523e-9297-aecf5f8bb48f:train:0/sampler_weights/final" \
        --output ./checkpoints/

    # Dry run (show download URL without downloading)
    uv run python -m scripts.download_checkpoint \
        --checkpoint "tinker://..." \
        --dry-run

Environment Variables:
    TINKER_API_KEY: Required for Tinker API access
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download checkpoint weights from Tinker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Tinker checkpoint path (e.g., tinker://...sampler_weights/final)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./checkpoints",
        help="Output directory for extracted weights (default: ./checkpoints)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded .tar archive after extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show download URL without downloading",
    )

    return parser.parse_args()


def download_checkpoint(
    checkpoint_path: str,
    output_dir: Path,
    keep_archive: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """
    Download and extract checkpoint from Tinker.

    Args:
        checkpoint_path: Tinker checkpoint path (tinker://...)
        output_dir: Directory to extract weights to
        keep_archive: Whether to keep the .tar archive after extraction
        dry_run: If True, only show the download URL

    Returns:
        Path to the extracted checkpoint directory, or None if dry run
    """
    # Import tinker SDK for REST client
    try:
        from tinker import ServiceClient
    except ImportError:
        console.print("[red]✗ tinker SDK not installed[/]")
        console.print("  Install with: uv pip install tinker")
        sys.exit(1)

    # Validate checkpoint path format
    if not checkpoint_path.startswith("tinker://"):
        console.print(f"[red]✗ Invalid checkpoint path: {checkpoint_path}[/]")
        console.print("  Path must start with 'tinker://'")
        sys.exit(1)

    console.print(f"\n[bold]Checkpoint:[/] {checkpoint_path}")

    # Initialize Tinker REST client
    console.print("\n[bold blue]Connecting to Tinker...[/]")
    try:
        sc = ServiceClient()
        rc = sc.create_rest_client()
        console.print("  ✓ Connected to Tinker")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to Tinker: {e}[/]")
        sys.exit(1)

    # Get signed download URL
    console.print("\n[bold blue]Getting download URL...[/]")
    try:
        future = rc.get_checkpoint_archive_url_from_tinker_path(checkpoint_path)
        response = future.result()
        download_url = response.url
        console.print("  ✓ Got signed download URL")
    except Exception as e:
        console.print(f"[red]✗ Failed to get download URL: {e}[/]")
        sys.exit(1)

    if dry_run:
        console.print(f"\n[yellow]Dry run - Download URL:[/]")
        console.print(f"  {download_url[:100]}...")
        return None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract checkpoint name from path for archive filename
    # e.g., "tinker://uuid:train:0/sampler_weights/final" -> "final"
    checkpoint_name = checkpoint_path.split("/")[-1]
    archive_path = output_dir / f"{checkpoint_name}.tar"

    # Download the archive
    console.print(f"\n[bold blue]Downloading checkpoint...[/]")
    console.print(f"  Archive: {archive_path}")
    try:
        urllib.request.urlretrieve(download_url, archive_path)
        archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
        console.print(f"  ✓ Downloaded ({archive_size_mb:.1f} MB)")
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/]")
        sys.exit(1)

    # Extract the archive
    console.print(f"\n[bold blue]Extracting checkpoint...[/]")
    extract_dir = output_dir / checkpoint_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=extract_dir)
        console.print(f"  ✓ Extracted to: {extract_dir}")
    except Exception as e:
        console.print(f"[red]✗ Extraction failed: {e}[/]")
        sys.exit(1)

    # List extracted files
    extracted_files = list(extract_dir.rglob("*"))
    file_count = len([f for f in extracted_files if f.is_file()])
    console.print(f"  Files: {file_count}")

    # Clean up archive unless --keep-archive
    if not keep_archive:
        archive_path.unlink()
        console.print(f"  ✓ Removed archive")

    console.print(f"\n[green]✓ Checkpoint downloaded successfully![/]")
    console.print(f"  Location: {extract_dir}")

    return extract_dir


def main() -> int:
    """Entry point."""
    # Load environment variables
    load_dotenv()

    args = parse_args()

    # Check for TINKER_API_KEY
    if not os.getenv("TINKER_API_KEY"):
        console.print("[red]✗ TINKER_API_KEY not set[/]")
        console.print("  Set it in .env or export it as an environment variable")
        return 1

    output_dir = Path(args.output)

    download_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        keep_archive=args.keep_archive,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

