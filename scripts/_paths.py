from pathlib import Path
import os


def project_root() -> Path:
    override = os.environ.get("GPT2_PIANO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
