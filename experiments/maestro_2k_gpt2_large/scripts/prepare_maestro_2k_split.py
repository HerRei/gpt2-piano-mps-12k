import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "data" / "raw" / "maestro-v3.0.0"
SPLITS_ROOT = EXPERIMENT_ROOT / "data" / "splits"
MANIFEST_PATH = EXPERIMENT_ROOT / "data" / "split_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 2k-piece MAESTRO subset with train/validation/test splits."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory of the MAESTRO dataset.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=2000,
        help="Total number of pieces to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize selected files into the split directories.",
    )
    return parser.parse_args()


def find_metadata_csv(source_root: Path) -> Path | None:
    candidates = sorted(source_root.glob("maestro*.csv"))
    return candidates[0] if candidates else None


def sanitize_split_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized == "validation":
        return "validation"
    if normalized == "test":
        return "test"
    return "train"


def allocate_counts(total_target: int, counts: dict[str, int]) -> dict[str, int]:
    total_available = sum(counts.values())
    if total_available < total_target:
        raise ValueError(f"requested {total_target} files but only found {total_available}")

    exact = {
        split: (total_target * count / total_available) if total_available else 0.0
        for split, count in counts.items()
    }
    allocated = {
        split: min(counts[split], int(math.floor(value)))
        for split, value in exact.items()
    }
    remaining = total_target - sum(allocated.values())

    if remaining > 0:
        remainders = sorted(
            counts.keys(),
            key=lambda split: (exact[split] - math.floor(exact[split]), counts[split]),
            reverse=True,
        )
        for split in remainders:
            if remaining == 0:
                break
            room = counts[split] - allocated[split]
            if room <= 0:
                continue
            take = min(room, remaining)
            allocated[split] += take
            remaining -= take

    if remaining != 0:
        raise ValueError("failed to allocate the requested subset size")

    return allocated


def build_split_map_from_metadata(source_root: Path, metadata_csv: Path) -> dict[str, list[Path]]:
    split_map = {"train": [], "validation": [], "test": []}

    with metadata_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "midi_filename" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"metadata csv missing required columns: {metadata_csv}")

        for row in reader:
            rel_path = row["midi_filename"].strip()
            split = sanitize_split_name(row["split"])
            src = source_root / rel_path
            if src.is_file():
                split_map[split].append(src)

    return split_map


def build_split_map_without_metadata(source_root: Path, seed: int) -> dict[str, list[Path]]:
    files = sorted(list(source_root.rglob("*.mid")) + list(source_root.rglob("*.midi")))
    files = [path for path in files if path.is_file()]
    if not files:
        raise ValueError(f"no MIDI files found in {source_root}")

    rng = random.Random(seed)
    rng.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * 0.90)
    n_val = int(n_total * 0.05)
    n_test = n_total - n_train - n_val

    return {
        "train": files[:n_train],
        "validation": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:n_train + n_val + n_test],
    }


def clear_split_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()


def materialize_split(split: str, files: list[Path], mode: str):
    out_dir = SPLITS_ROOT / split
    clear_split_dir(out_dir)

    for idx, src in enumerate(files):
        ext = src.suffix.lower()
        dst = out_dir / f"{split}_{idx:05d}{ext}"
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    metadata_csv = find_metadata_csv(source_root)
    if metadata_csv is not None:
        split_map = build_split_map_from_metadata(source_root, metadata_csv)
        split_source = "metadata_csv"
    else:
        split_map = build_split_map_without_metadata(source_root, args.seed)
        split_source = "random_fallback"

    counts = {split: len(files) for split, files in split_map.items()}
    allocated = allocate_counts(args.subset_size, counts)

    rng = random.Random(args.seed)
    selected = {}
    for split, files in split_map.items():
        chosen = list(files)
        rng.shuffle(chosen)
        chosen = sorted(chosen[:allocated[split]])
        selected[split] = chosen
        materialize_split(split, chosen, args.link_mode)

    manifest = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "source_root": str(source_root),
        "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
        "split_source": split_source,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "link_mode": args.link_mode,
        "available_counts": counts,
        "selected_counts": {split: len(files) for split, files in selected.items()},
        "selected_files": {
            split: [str(path.relative_to(source_root)) for path in files]
            for split, files in selected.items()
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("prepared maestro 2k subset", flush=True)
    print(json.dumps(manifest["selected_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
