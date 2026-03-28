import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from _paths import ROOT
from pipeline_utils import (
    build_generation_command,
    checkpoint_dir,
    load_prompt_profiles,
    normalize_epoch,
    parse_csv,
)


DEFAULT_PROFILES = [
    "schubert_lyrical",
    "recital_compact",
    "schubert_bright",
    "recital_expansive",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a batch generation pipeline for the epoch 2 and epoch 4 checkpoints."
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="2,4",
        help="Comma-separated epoch numbers or checkpoint labels like 2,4 or epoch_02,epoch_04.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated prompt profile names from configs/generation_prompt_profiles.json.",
    )
    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=ROOT / "configs" / "generation_prompt_profiles.json",
        help="JSON file containing named prompt profiles.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch generate_piano_sample.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory. Defaults to exports/pipeline_runs/<timestamp>.",
    )
    parser.add_argument(
        "--prompt-midi",
        type=Path,
        default=None,
        help="Optional custom seed MIDI applied to every pipeline job.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=Path,
        default=None,
        help="Optional custom .npy token prompt applied to every pipeline job.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Optional override for num candidates per job.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base random seed. Each profile adds its own offset and each job adds its index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write the manifest without running generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prompt_midi is not None and args.prompt_tokens is not None:
        raise ValueError("use only one of --prompt-midi or --prompt-tokens")

    profiles_file = args.profiles_file.expanduser().resolve()
    profiles = load_prompt_profiles(profiles_file)

    requested_profile_names = parse_csv(args.profiles) or DEFAULT_PROFILES
    selected_profiles = []
    for profile_name in requested_profile_names:
        if profile_name not in profiles:
            raise KeyError(f"unknown prompt profile: {profile_name}")
        selected_profiles.append(profiles[profile_name])

    epochs = [normalize_epoch(epoch) for epoch in parse_csv(args.epochs)]
    if not epochs:
        raise ValueError("at least one epoch is required")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "exports" / "pipeline_runs" / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "profiles_file": str(profiles_file),
        "epochs": epochs,
        "profile_names": requested_profile_names,
        "dry_run": bool(args.dry_run),
        "jobs": [],
    }

    job_counter = 0
    for epoch in epochs:
        checkpoint = checkpoint_dir(ROOT, epoch)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        epoch_output_dir = output_dir / f"epoch_{epoch:02d}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

        for profile in selected_profiles:
            job_counter += 1
            seed = args.seed_base + profile.seed_offset + job_counter
            run_name = f"epoch{epoch:02d}_{profile.name}"
            command = build_generation_command(
                python_bin=args.python_bin,
                root=ROOT,
                checkpoint=checkpoint,
                output_dir=epoch_output_dir,
                run_name=run_name,
                profile=profile,
                seed=seed,
                prompt_midi=args.prompt_midi.expanduser().resolve() if args.prompt_midi else None,
                prompt_tokens=args.prompt_tokens.expanduser().resolve() if args.prompt_tokens else None,
                num_candidates_override=args.num_candidates,
            )

            job = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "profile": profile.name,
                "description": profile.description,
                "seed": seed,
                "output_dir": str(epoch_output_dir),
                "command": command,
                "command_shell": " ".join(shlex.quote(part) for part in command),
                "status": "planned",
            }
            manifest["jobs"].append(job)

    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for job in manifest["jobs"]:
        print(job["command_shell"], flush=True)
        if args.dry_run:
            job["status"] = "dry_run"
            continue

        subprocess.run(job["command"], cwd=ROOT, check=True)
        job["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
