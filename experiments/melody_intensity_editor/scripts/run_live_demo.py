import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_DEMOS = EXPERIMENT_ROOT / "examples" / "demo_songs.json"
DEFAULT_CHECKPOINT = EXPERIMENT_ROOT / "checkpoints" / "best"
DEFAULT_MAESTRO_ROOT = Path(
    os.environ.get("MAESTRO_ROOT", str(WORKSPACE_ROOT / "data" / "raw" / "maestro-v3.0.0"))
)
DEFAULT_PYTHON_BIN = Path(
    os.environ.get("PYTHON_BIN", sys.executable)
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a named live-test demo using a local MAESTRO MIDI example."
    )
    parser.add_argument("--list", action="store_true", help="List the available demos and exit.")
    parser.add_argument("--demo-key", type=str, default=None, help="Which named demo to run.")
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--maestro-root", type=Path, default=DEFAULT_MAESTRO_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def load_demo_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"demo manifest not found: {path}")
    return json.loads(path.read_text())


def resolve_python_bin(path: Path) -> Path:
    raw = str(path)
    if os.sep not in raw:
        resolved = shutil.which(raw)
        if resolved is not None:
            return Path(resolved).resolve()
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"python bin not found: {path}")


def print_demos(payload: dict):
    default_demo = payload.get("default_demo")
    for key, demo in payload["demos"].items():
        marker = " (default)" if key == default_demo else ""
        print(f"{key}{marker}", flush=True)
        print(f"  {demo['composer']} - {demo['title']}", flush=True)
        print(f"  {demo['description']}", flush=True)


def main():
    args, passthrough = parse_args()
    payload = load_demo_config(args.demos.expanduser().resolve())

    if args.list:
        print_demos(payload)
        return

    demo_key = args.demo_key or payload.get("default_demo")
    if demo_key not in payload["demos"]:
        available = ", ".join(sorted(payload["demos"]))
        raise KeyError(f"unknown demo-key '{demo_key}'. Available demos: {available}")

    demo = payload["demos"][demo_key]
    maestro_root = args.maestro_root.expanduser().resolve()
    input_midi = maestro_root / demo["maestro_relpath"]
    if not input_midi.exists():
        raise FileNotFoundError(
            f"demo MIDI not found: {input_midi}\n"
            f"Set --maestro-root to your MAESTRO v3.0.0 folder."
        )

    python_bin = resolve_python_bin(args.python_bin)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    cmd = [
        str(python_bin),
        str(EXPERIMENT_ROOT / "scripts" / "test_live_intensity_pipeline.py"),
        "--checkpoint",
        str(checkpoint),
        "--input-midi",
        str(input_midi),
        "--window-beats",
        str(demo["window_beats"]),
        "--hop-beats",
        str(demo["hop_beats"]),
        "--schedule-mode",
        str(demo["schedule_mode"]),
        "--intensity-schedule",
        str(demo["intensity_schedule"]),
        "--name",
        str(demo["suggested_name"]),
    ]
    if demo.get("extract_melody", False):
        cmd.append("--extract-melody")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(passthrough)

    print(f"[demo] key={demo_key}", flush=True)
    print(f"[demo] title={demo['composer']} - {demo['title']}", flush=True)
    print(f"[demo] midi={input_midi}", flush=True)
    print(f"[demo] checkpoint={checkpoint}", flush=True)
    print(f"[demo] schedule={demo['intensity_schedule']}", flush=True)
    print(f"[demo] command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
