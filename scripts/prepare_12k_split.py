from pathlib import Path
import random
import shutil

from _paths import ROOT

SRC = ROOT / "data" / "raw" / "source_midis"
SPLITS = ROOT / "data" / "splits"

TRAIN_DIR = SPLITS / "train"
VAL_DIR = SPLITS / "validation"
TEST_DIR = SPLITS / "test"

SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

random.seed(SEED)

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

files = sorted(list(SRC.rglob("*.mid")) + list(SRC.rglob("*.midi")))
files = [f for f in files if f.is_file()]

if not files:
    raise SystemExit(f"No MIDI files found in {SRC}")

random.shuffle(files)

n = len(files)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_files = files[:n_train]
val_files = files[n_train:n_train + n_val]
test_files = files[n_train + n_val:]


def link_or_copy(src_files, out_dir, prefix):
    for old in out_dir.glob("*"):
        if old.is_symlink() or old.is_file():
            old.unlink()

    for i, src in enumerate(src_files):
        ext = src.suffix.lower()
        dst = out_dir / f"{prefix}_{i:05d}{ext}"
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


link_or_copy(train_files, TRAIN_DIR, "train")
link_or_copy(val_files, VAL_DIR, "val")
link_or_copy(test_files, TEST_DIR, "test")

print("done")
print(f"total: {n}")
print(f"train: {len(train_files)}")
print(f"validation: {len(val_files)}")
print(f"test: {len(test_files)}")
