"""
download_data.py — Fetch the NASA C-MAPSS FD001 dataset.

Downloads from a public mirror and extracts the required files into ./data/.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path("data")
# Public mirror on GitHub (widely cited in academic repos)
DATASET_URL = (
    "https://raw.githubusercontent.com/makinarocks/awesome-industrial-machine-datasets"
    "/master/data-list/NASA-CMAPS/CMAPSSData.zip"
)
REQUIRED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
ZIP_PATH = DATA_DIR / "CMAPSSData.zip"


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    if all((DATA_DIR / f).exists() for f in REQUIRED_FILES):
        print("✓ FD001 dataset files already present in ./data/")
        return

    print(f"Downloading C-MAPSS dataset from:\n  {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    except Exception as e:
        print(
            f"\n✗ Download failed: {e}\n\n"
            "Please manually download the C-MAPSS dataset and place\n"
            "the following files in the ./data/ directory:\n"
            "  • train_FD001.txt\n"
            "  • test_FD001.txt\n"
            "  • RUL_FD001.txt\n\n"
            "Sources:\n"
            "  • https://www.kaggle.com/datasets/samirshr/cmapss-jet-engine-simulated-data\n"
            "  • NASA Prognostics Data Repository\n"
        )
        sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if basename in REQUIRED_FILES:
                # Extract flat into data/
                with zf.open(member) as src, open(DATA_DIR / basename, "wb") as dst:
                    dst.write(src.read())
                print(f"  → {basename}")

    ZIP_PATH.unlink(missing_ok=True)

    # Final check
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print(f"✗ Missing files after extraction: {missing}")
        sys.exit(1)

    print("✓ FD001 dataset ready in ./data/")


if __name__ == "__main__":
    download_and_extract()
