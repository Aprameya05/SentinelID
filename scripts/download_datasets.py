"""
Dataset download and preparation script.

Handles downloading and extracting all datasets used by SentinelID.
Some datasets (MS-Celeb-1M, DISFA) require manual registration.
This script handles the rest automatically.
"""

import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn

console = Console()

DATASET_INFO = {
    "faceforensics": {
        "description": "FaceForensics++ (4 manipulation types + original)",
        "url": "https://github.com/ondyari/FaceForensics",
        "manual": True,
        "note": "Requires signing a request form at the GitHub repo. Run their download script.",
        "size": "~1.3TB (compressed video)",
        "target_dir": "data/raw/faceforensics",
    },
    "celebdf": {
        "description": "CelebDF-v2 (high-quality deepfake dataset)",
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "manual": True,
        "note": "Email request to authors as per their GitHub instructions.",
        "size": "~2.1GB",
        "target_dir": "data/raw/celebdf_v2",
    },
    "nuaa": {
        "description": "NUAA Imposter Database (print attack liveness)",
        "url": "http://parnec.nuaa.edu.cn/xtan/data/nuaaimposter.html",
        "manual": True,
        "note": "Request access via the NUAA website linked above.",
        "size": "~1.2GB",
        "target_dir": "data/raw/nuaa",
    },
    "vggface2": {
        "description": "VGGFace2 (9K identities, 3.3M images)",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/",
        "manual": True,
        "note": "Download requires signing the academic license at Oxford VGG.",
        "size": "~36GB",
        "target_dir": "data/raw/vggface2",
    },
    "midv500": {
        "description": "MIDV-500 (identity document dataset)",
        "url": "http://smartengines.com/midv-500",
        "manual": True,
        "note": "Available via Smart Engines research page.",
        "size": "~4.8GB",
        "target_dir": "data/raw/midv500",
    },
    "lfw": {
        "description": "Labeled Faces in the Wild (face verification benchmark)",
        "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "manual": False,
        "size": "~170MB",
        "target_dir": "data/raw/lfw",
    },
    "disfa": {
        "description": "Denver Intensity of Spontaneous Facial Action (AU detection)",
        "url": "http://mohammadmahoor.com/disfa/",
        "manual": True,
        "note": "Request access via the DISFA website (academic license required).",
        "size": "~66GB",
        "target_dir": "data/raw/disfa",
    },
}


def print_dataset_table():
    from rich.table import Table
    table = Table(title="SentinelID Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Size", justify="right")
    table.add_column("Auto?", justify="center")
    table.add_column("Note")

    for name, info in DATASET_INFO.items():
        auto = "[green]Yes[/green]" if not info["manual"] else "[red]No[/red]"
        note = info.get("note", "")[:60]
        table.add_row(name, info["description"], info["size"], auto, note)

    console.print(table)


def download_lfw(target_dir: Path):
    """Download LFW dataset (no auth required)."""
    import urllib.request
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir.parent / "lfw.tgz"

    if (target_dir / "Aaron_Eckhart").exists():
        console.print("[dim]LFW already downloaded.[/dim]")
        return

    console.print(f"Downloading LFW from {url}...")
    urllib.request.urlretrieve(url, archive)

    console.print("Extracting LFW...")
    with tarfile.open(archive) as tf:
        tf.extractall(target_dir.parent)

    archive.unlink()
    console.print("[green]LFW ready.[/green]")


def main():
    parser = argparse.ArgumentParser(description="Download SentinelID datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_INFO.keys()) + ["all"],
        default=["lfw"],
        help="Which datasets to download",
    )
    parser.add_argument("--list", action="store_true", help="List all datasets and exit")
    parser.add_argument("--data-root", default="data/raw")
    args = parser.parse_args()

    if args.list:
        print_dataset_table()
        return

    targets = list(DATASET_INFO.keys()) if "all" in args.datasets else args.datasets

    for name in targets:
        info = DATASET_INFO[name]
        console.rule(f"[bold]{name}[/bold]")
        console.print(f"  {info['description']}")

        if info["manual"]:
            console.print(f"  [yellow]Manual download required:[/yellow]")
            console.print(f"  URL:  {info['url']}")
            console.print(f"  Note: {info.get('note', '')}")
            console.print(f"  Place data in: {info['target_dir']}")
        else:
            target = Path(args.data_root) / Path(info["target_dir"]).name
            if name == "lfw":
                download_lfw(target)

    console.print("\n[green]Done.[/green]")
    print_dataset_table()


if __name__ == "__main__":
    main()
