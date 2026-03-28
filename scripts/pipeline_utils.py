from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass(frozen=True)
class PromptProfile:
    name: str
    description: str
    prompt_split: str
    prompt_index: int
    prompt_position: str
    prompt_length: int
    max_new_tokens: int
    preset: str
    num_candidates: int
    cpu_threads: int
    seed_offset: int = 0


def parse_csv(spec: Optional[str]) -> List[str]:
    if spec is None:
        return []
    return [chunk.strip() for chunk in spec.split(",") if chunk.strip()]


def normalize_epoch(raw_epoch) -> int:
    if isinstance(raw_epoch, int):
        if raw_epoch < 0:
            raise ValueError("epoch must be non-negative")
        return raw_epoch

    text = str(raw_epoch).strip().lower()
    if text.startswith("epoch_"):
        text = text.split("_", 1)[1]
    epoch = int(text)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return epoch


def checkpoint_dir(root: Path, epoch: int) -> Path:
    return root / "checkpoints" / f"epoch_{epoch:02d}"


def load_prompt_profiles(path: Path) -> Dict[str, PromptProfile]:
    payload = json.loads(path.read_text())
    profiles = {}
    for name, config in payload.items():
        profiles[name] = PromptProfile(
            name=name,
            description=config["description"],
            prompt_split=config["prompt_split"],
            prompt_index=int(config["prompt_index"]),
            prompt_position=config["prompt_position"],
            prompt_length=int(config["prompt_length"]),
            max_new_tokens=int(config["max_new_tokens"]),
            preset=config["preset"],
            num_candidates=int(config["num_candidates"]),
            cpu_threads=int(config["cpu_threads"]),
            seed_offset=int(config.get("seed_offset", 0)),
        )
    return profiles


def build_generation_command(
    python_bin: str,
    root: Path,
    checkpoint: Path,
    output_dir: Path,
    run_name: str,
    profile: PromptProfile,
    seed: int,
    prompt_midi: Optional[Path] = None,
    prompt_tokens: Optional[Path] = None,
    num_candidates_override: Optional[int] = None,
) -> List[str]:
    command = [
        python_bin,
        str(root / "scripts" / "generate_piano_sample.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--name",
        run_name,
        "--prompt-length",
        str(profile.prompt_length),
        "--max-new-tokens",
        str(profile.max_new_tokens),
        "--preset",
        profile.preset,
        "--cpu-threads",
        str(profile.cpu_threads),
        "--seed",
        str(seed),
        "--num-candidates",
        str(num_candidates_override or profile.num_candidates),
    ]

    if prompt_midi is not None:
        command.extend(["--prompt-midi", str(prompt_midi)])
    elif prompt_tokens is not None:
        command.extend(["--prompt-tokens", str(prompt_tokens)])
    else:
        command.extend(
            [
                "--prompt-split",
                profile.prompt_split,
                "--prompt-index",
                str(profile.prompt_index),
                "--prompt-position",
                profile.prompt_position,
            ]
        )

    return command
