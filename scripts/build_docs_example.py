import argparse
import json
import shutil
from pathlib import Path

import pretty_midi

from _paths import ROOT


DEFAULT_MIDI = ROOT / "exports" / "generated" / "docs_search_best_balanced_focus_p01_middle_c06.mid"
DEFAULT_META = ROOT / "exports" / "generated" / "docs_search_best_balanced_focus_p01_middle_c06.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the GitHub Pages example assets from a generated MIDI sample."
    )
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def choose_instrument(pm: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    if not pm.instruments:
        raise ValueError("MIDI file has no instruments")
    return max(pm.instruments, key=lambda inst: len(inst.notes))


def note_payload(note: pretty_midi.Note) -> dict:
    return {
        "pitch": int(note.pitch),
        "velocity": int(note.velocity),
        "start": round(float(note.start), 6),
        "end": round(float(note.end), 6),
    }


def main():
    args = parse_args()

    midi_path = args.midi.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text())
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    instrument = choose_instrument(pm)

    tempo_times, tempi = pm.get_tempo_changes()
    tempo_bpm = float(tempi[0]) if len(tempi) else float(pm.estimate_tempo())
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch, note.end))

    payload = {
        "title": "Generated piano example",
        "description": "Best-scoring balanced-preset continuation from the current best checkpoint.",
        "duration_sec": round(float(pm.get_end_time()), 6),
        "tempo_bpm": round(tempo_bpm, 3),
        "track_name": instrument.name or "Piano",
        "note_count": len(notes),
        "pitch_range": {
            "min": int(min(note.pitch for note in notes)),
            "max": int(max(note.pitch for note in notes)),
        },
        "meta": {
            "checkpoint": metadata["checkpoint"],
            "preset": metadata["preset"],
            "seed": int(metadata["seed"]),
            "prompt_index": metadata["prompt_index"],
            "prompt_position": metadata["prompt_position"],
            "prompt_tokens": int(metadata["prompt_tokens"]),
            "generated_tokens": int(metadata["generated_tokens"]),
            "heuristic_score": float(metadata["heuristic_score"]),
            "note_count": int(metadata["note_count"]),
            "notes_per_beat": float(metadata["notes_per_beat"]),
            "pitch_span": int(metadata["pitch_span"]),
            "unique_pitches": int(metadata["unique_pitches"]),
            "velocity_std": float(metadata["velocity_std"]),
        },
        "notes": [note_payload(note) for note in notes],
    }

    shutil.copy2(midi_path, output_dir / "generated-example.mid")
    (output_dir / "generated-example.json").write_text(json.dumps(payload, indent=2))

    print(output_dir / "generated-example.mid")
    print(output_dir / "generated-example.json")


if __name__ == "__main__":
    main()
