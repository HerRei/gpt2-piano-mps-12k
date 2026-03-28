import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from _paths import ROOT
from pipeline_utils import (
    PipelineJob,
    build_generation_command,
    checkpoint_dir,
    format_profile_listing,
    load_prompt_profiles,
    normalize_epoch,
    parse_csv,
    select_profile_names,
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
        help="Comma-separated prompt profile names, or 'all'.",
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
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the available prompt profiles and exit.",
    )
    return parser.parse_args()


def resolve_output_dir(requested_dir: Optional[Path]) -> Path:
    if requested_dir is not None:
        return requested_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROOT / "exports" / "pipeline_runs" / timestamp).resolve()


def build_jobs(args, profiles_file: Path, profiles) -> tuple[list[str], list[PipelineJob], Path]:
    requested_profile_names = select_profile_names(args.profiles, sorted(profiles))
    selected_profiles = []
    for profile_name in requested_profile_names:
        if profile_name not in profiles:
            raise KeyError(f"unknown prompt profile: {profile_name}")
        selected_profiles.append(profiles[profile_name])

    epochs = [normalize_epoch(epoch) for epoch in parse_csv(args.epochs)]
    if not epochs:
        raise ValueError("at least one epoch is required")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[PipelineJob] = []
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
            jobs.append(
                PipelineJob(
                    epoch=epoch,
                    checkpoint=checkpoint,
                    output_dir=epoch_output_dir,
                    profile=profile,
                    seed=seed,
                    command=command,
                )
            )

    return requested_profile_names, jobs, output_dir


def build_manifest(args, profiles_file: Path, profile_names: list[str], jobs: list[PipelineJob]) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "profiles_file": str(profiles_file),
        "epochs": sorted({job.epoch for job in jobs}),
        "profile_names": profile_names,
        "dry_run": bool(args.dry_run),
        "jobs": [job.to_manifest_record() for job in jobs],
    }


def print_job_plan(jobs: list[PipelineJob], dry_run: bool) -> None:
    mode = "dry run" if dry_run else "execution"
    print(f"Pipeline mode: {mode}", flush=True)
    print(f"Jobs queued: {len(jobs)}", flush=True)
    for job in jobs:
        print(
            f"- epoch_{job.epoch:02d} | {job.profile.name} | seed={job.seed} | "
            f"output={job.output_dir}",
            flush=True,
        )


def main():
    args = parse_args()

    if args.prompt_midi is not None and args.prompt_tokens is not None:
        raise ValueError("use only one of --prompt-midi or --prompt-tokens")

    profiles_file = args.profiles_file.expanduser().resolve()
    profiles = load_prompt_profiles(profiles_file)

    if args.list_profiles:
        print(format_profile_listing(profiles))
        return

    requested_profile_names, jobs, output_dir = build_jobs(args, profiles_file, profiles)
    manifest = build_manifest(args, profiles_file, requested_profile_names, jobs)
    print_job_plan(jobs, dry_run=args.dry_run)

    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for index, job in enumerate(jobs):
        print(" ".join(shlex.quote(part) for part in job.command), flush=True)
        if args.dry_run:
            manifest["jobs"][index]["status"] = "dry_run"
            continue

        subprocess.run(job.command, cwd=ROOT, check=True)
        manifest["jobs"][index]["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
