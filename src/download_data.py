"""Download the RIDE Gold Lite dataset (core + tabular tiers) from Hugging Face.

Only the files needed by this pipeline are fetched:
  - core/            benchmark spec, metadata, and the canonical test evaluation table
  - tabular/         dense tabular arrays used by the XGBoost baseline (train + test)

The download is idempotent: already-downloaded files are skipped by the
huggingface_hub cache. Total size is roughly 7.3 GB, dominated by
tabular/train/x.npy (5.7 GB).

Usage:
    python src/download_data.py [--schema-only]
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "orailix/ride-gold-lite"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Small files: schemas, specs, and the 74 MB evaluation table.
SCHEMA_PATTERNS = [
    "README.md",
    "core/*",
    "tabular/*.yaml",
]

# Large dense arrays (~7 GB total).
ARRAY_PATTERNS = [
    "tabular/train/*",
    "tabular/test/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only download schemas, specs, and the test evaluation table.",
    )
    args = parser.parse_args()

    patterns = SCHEMA_PATTERNS if args.schema_only else SCHEMA_PATTERNS + ARRAY_PATTERNS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=patterns,
    )
    print(f"Downloaded RIDE Gold Lite files to: {path}")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            print(f"  {f.relative_to(DATA_DIR)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
