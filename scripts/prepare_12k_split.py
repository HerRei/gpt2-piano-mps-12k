from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from _paths import ROOT

DEFAULT_SOURCE_DIR = ROOT / "data" / "raw" / "source_midis"
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_VAL_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05
LEAKAGE_NOTE = (
    "This is a shuffled file-level split. It does not deduplicate alternate takes, "
    "near-duplicates, or related arrangements. If the source corpus contains linked "
    "performances, group them before splitting to avoid leakage across train, "
    "validation, and test."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a random file-level train/validation/test split for the 12k MIDI corpus."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing raw source MIDIs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory that will receive the split folders.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks with copy fallback.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float):
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")

    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")


def link_or_copy(src_files, out_dir: Path, prefix: str, force_copy: bool):
    for old in out_dir.glob("*"):
        if old.is_symlink() or old.is_file():
            old.unlink()

    for index, src in enumerate(src_files):
        ext = src.suffix.lower()
        dst = out_dir / f"{prefix}_{index:05d}{ext}"
        if force_copy:
            shutil.copy2(src, dst)
            continue

        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    test_dir = output_dir / "test"

    random.seed(args.seed)

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(source_dir.rglob("*.mid")) + list(source_dir.rglob("*.midi")))
    files = [path for path in files if path.is_file()]

    if not files:
        raise SystemExit(f"No MIDI files found in {source_dir}")

    random.shuffle(files)

    total_files = len(files)
    n_train = int(total_files * args.train_ratio)
    n_val = int(total_files * args.validation_ratio)
    n_test = total_files - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    link_or_copy(train_files, train_dir, "train", force_copy=args.copy)
    link_or_copy(val_files, val_dir, "val", force_copy=args.copy)
    link_or_copy(test_files, test_dir, "test", force_copy=args.copy)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "strategy": "random_file_shuffle",
        "counts": {
            "total": total_files,
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        },
        "copy_mode": bool(args.copy),
        "leakage_note": LEAKAGE_NOTE,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("done")
    print(f"total: {total_files}")
    print(f"train: {len(train_files)}")
    print(f"validation: {len(val_files)}")
    print(f"test: {len(test_files)}")
    print(f"note: {LEAKAGE_NOTE}")


if __name__ == "__main__":
    main()
