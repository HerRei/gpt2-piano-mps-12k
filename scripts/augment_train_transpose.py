from pathlib import Path
import shutil
import pretty_midi

from _paths import ROOT

TRAIN_SRC = ROOT / "data" / "splits" / "train"
TRAIN_AUG = ROOT / "data" / "augmented" / "train"

TRAIN_AUG.mkdir(parents=True, exist_ok=True)

# Safe musical transpositions for piano dataset expansion
TRANSPOSES = [-3, -2, -1, 1, 2, 3]

PIANO_LOW = 21
PIANO_HIGH = 108


def midi_note_range(pm: pretty_midi.PrettyMIDI):
    notes = []
    for inst in pm.instruments:
        for note in inst.notes:
            notes.append(note.pitch)
    if not notes:
        return None, None
    return min(notes), max(notes)


def transpose_midi(src: Path, dst: Path, semitones: int) -> bool:
    pm = pretty_midi.PrettyMIDI(str(src))
    lo, hi = midi_note_range(pm)
    if lo is None:
        return False

    if lo + semitones < PIANO_LOW or hi + semitones > PIANO_HIGH:
        return False

    for inst in pm.instruments:
        for note in inst.notes:
            note.pitch += semitones

    pm.write(str(dst))
    return True


files = sorted(list(TRAIN_SRC.glob("*.mid")) + list(TRAIN_SRC.glob("*.midi")))
if not files:
    raise SystemExit(f"No train MIDI files found in {TRAIN_SRC}")

# Clean old augmented files first
for old in TRAIN_AUG.glob("*"):
    if old.is_symlink() or old.is_file():
        old.unlink()

copied_original = 0
created_aug = 0
skipped_aug = 0

for i, src in enumerate(files):
    ext = src.suffix.lower()
    stem = src.stem

    # keep original in augmented training set
    dst_orig = TRAIN_AUG / f"{stem}_orig{ext}"
    try:
        dst_orig.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst_orig)
    copied_original += 1

    # transposed copies
    for st in TRANSPOSES:
        sign = "p" if st > 0 else "m"
        mag = abs(st)
        dst = TRAIN_AUG / f"{stem}_tr_{sign}{mag}{ext}"
        ok = transpose_midi(src, dst, st)
        if ok:
            created_aug += 1
        else:
            skipped_aug += 1

    if (i + 1) % 100 == 0:
        print(f"processed {i+1}/{len(files)}")

total_aug_files = len(list(TRAIN_AUG.glob("*.mid")) + list(TRAIN_AUG.glob("*.midi")))

print("done")
print(f"original copied: {copied_original}")
print(f"augmented created: {created_aug}")
print(f"augmented skipped: {skipped_aug}")
print(f"total files in augmented/train: {total_aug_files}")
