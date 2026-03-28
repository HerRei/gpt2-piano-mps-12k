from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from _paths import ROOT


ART = ROOT / "artifacts"
DEFAULT_TOKENS_DIR = ART / "tokens"
DEFAULT_META_PATH = ART / "meta.json"
DEFAULT_CHECKPOINTS_DIR = ROOT / "checkpoints"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_RESUME_FROM = DEFAULT_CHECKPOINTS_DIR / "epoch_01"
TRAINER_STATE_NAME = "trainer_state.pt"

DEFAULT_BLOCK_SIZE = 2048
DEFAULT_STRIDE = 1024
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42
DEFAULT_LOG_EVERY_OPT_STEPS = 25

MODEL_CFG = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}


@dataclass
class RunPaths:
    tokens: Path
    meta: Path
    checkpoints: Path
    logs: Path

    @property
    def status(self) -> Path:
        return self.logs / "run_status.json"


@dataclass
class ResumeInfo:
    checkpoint: Path | None
    start_epoch: int = 1
    elapsed: float = 0.0
    best_val: float = field(default_factory=lambda: float("inf"))
    verification: dict | None = None
    note: dict = field(
        default_factory=lambda: {
            "mode": "fresh_start",
            "message": "starting from scratch",
        }
    )
    trainer_state: dict | None = None


@dataclass
class EpochSummary:
    train_loss: float
    val_loss: float
    epoch_time: float
    total_elapsed: float
    epochs_left: int
    eta_next: float
    global_step: int


class MidiWindowDataset(Dataset):
    def __init__(self, split_dir: Path, block_size: int, stride: int):
        self.block_size = block_size
        self.arrays = []
        self.index = []

        for path in sorted(split_dir.glob("*.npy")):
            arr = np.load(path, mmap_mode="r")
            if len(arr) < block_size + 1:
                continue

            arr_id = len(self.arrays)
            self.arrays.append(arr)

            last_start = len(arr) - (block_size + 1)
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                self.index.append((arr_id, start))

        print(
            f"{split_dir.name}: files={len(self.arrays)} windows={len(self.index)}",
            flush=True,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        arr_id, start = self.index[idx]
        arr = self.arrays[arr_id]
        window = np.asarray(arr[start:start + self.block_size], dtype=np.int64)
        return torch.tensor(window, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the main 12k symbolic piano GPT-2 model."
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=DEFAULT_TOKENS_DIR,
        help="Directory containing tokenized train/validation/test splits.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Path to artifacts/meta.json with the tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Directory used for epoch checkpoints and the best checkpoint.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory used for run status logs.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=str(DEFAULT_RESUME_FROM),
        help="Checkpoint directory to resume from, or 'none' to force a fresh run.",
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Ignore --resume-from and start with a freshly initialized model.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--log-every-opt-steps",
        type=int,
        default=DEFAULT_LOG_EVERY_OPT_STEPS,
        help="Print a progress line after this many optimizer steps.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.log_every_opt_steps < 1:
        raise ValueError("--log-every-opt-steps must be positive")


def fmt_seconds(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def fmt_progress_bar(progress: float, width: int = 16) -> str:
    progress = min(max(progress, 0.0), 1.0)
    if progress >= 1.0:
        return "#" * width

    filled = int(progress * width)
    if filled <= 0:
        return ">" + "." * (width - 1)
    return "#" * (filled - 1) + ">" + "." * (width - filled)


def write_status(status_path: Path, payload: dict):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2))


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def state_dict_fingerprint(state_dict: dict) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    tensor_count = 0
    param_count = 0

    for name in sorted(state_dict):
        tensor = state_dict[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor_bytes(tensor))
        tensor_count += 1
        param_count += tensor.numel()

    return hasher.hexdigest()[:16], tensor_count, param_count


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_resume_path(args) -> Path | None:
    if args.train_from_scratch:
        return None

    raw = (args.resume_from or "").strip()
    if not raw or raw.lower() in {"none", "scratch", "false"}:
        return None

    return Path(raw).expanduser().resolve()


def load_vocab_size(meta_path: Path) -> int:
    return int(json.loads(meta_path.read_text())["vocab_size"])


def model_shape(vocab_size: int, block_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "n_positions": block_size,
        "n_ctx": block_size,
        "n_embd": MODEL_CFG["n_embd"],
        "n_layer": MODEL_CFG["n_layer"],
        "n_head": MODEL_CFG["n_head"],
    }


def config_view(config_or_dict, expected: dict) -> dict:
    source = config_or_dict if isinstance(config_or_dict, dict) else config_or_dict.to_dict()
    return {key: source[key] for key in expected}


def check_model_config(config_or_dict, expected: dict, label: str):
    actual = config_view(config_or_dict, expected)
    mismatches = []

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

    if mismatches:
        raise RuntimeError(f"{label} config mismatch:\n" + "\n".join(mismatches))


def build_model_config(vocab_size: int, block_size: int) -> GPT2Config:
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=MODEL_CFG["n_embd"],
        n_layer=MODEL_CFG["n_layer"],
        n_head=MODEL_CFG["n_head"],
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
    )


def verify_resume_checkpoint(
    model: GPT2LMHeadModel,
    checkpoint: Path,
    expected: dict,
) -> dict:
    check_model_config(model.config, expected, "loaded model")
    check_model_config(
        json.loads((checkpoint / "config.json").read_text()),
        expected,
        f"{checkpoint / 'config.json'}",
    )

    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"resume model weights not found: {model_path}")

    try:
        from safetensors.torch import load_file
    except ImportError:
        return {
            "mode": "config_only",
            "message": "safetensors unavailable; verified config only",
        }

    saved_state = load_file(str(model_path))
    model_state = model.state_dict()

    missing = sorted(set(saved_state) - set(model_state))
    if missing:
        raise RuntimeError(f"loaded model is missing checkpoint tensor: {missing[0]}")

    for name in sorted(saved_state):
        expected_tensor = saved_state[name]
        actual_tensor = model_state[name].detach().cpu()

        if expected_tensor.shape != actual_tensor.shape:
            raise RuntimeError(
                f"resume tensor shape mismatch for {name}: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}"
            )

        if expected_tensor.dtype != actual_tensor.dtype:
            raise RuntimeError(
                f"resume tensor dtype mismatch for {name}: "
                f"expected {expected_tensor.dtype}, got {actual_tensor.dtype}"
            )

        if not torch.equal(expected_tensor, actual_tensor):
            diff = (
                expected_tensor.to(torch.float32) - actual_tensor.to(torch.float32)
            ).abs().max().item()
            raise RuntimeError(
                f"resume tensor value mismatch for {name}: max_abs_diff={diff}"
            )

    fingerprint, tensor_count, param_count = state_dict_fingerprint(saved_state)
    return {
        "mode": "full",
        "fingerprint": fingerprint,
        "tensor_count": tensor_count,
        "param_count": param_count,
    }


def run_config(args, vocab_size: int) -> dict:
    return {
        "block_size": args.block_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "model": model_shape(vocab_size, args.block_size),
    }


def check_resume_args(saved: dict | None, current: dict):
    if not saved:
        return

    keys = [
        "block_size",
        "stride",
        "batch_size",
        "grad_accum",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "seed",
        "vocab_size",
    ]
    mismatches = []

    for key in keys:
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: expected {saved.get(key)}, got {current.get(key)}")

    if mismatches:
        raise RuntimeError("resume training config mismatch:\n" + "\n".join(mismatches))


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def restore_rng_state(state: dict | None):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_mps" in state and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(state["torch_mps"])
        except RuntimeError:
            pass


def trainer_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / TRAINER_STATE_NAME


def load_torch_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_trainer_state(checkpoint: Path) -> dict | None:
    path = trainer_state_path(checkpoint)
    if not path.exists():
        return None
    payload = load_torch_payload(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid trainer state payload: {path}")
    return payload


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_trainer_state(
    *,
    epoch: int,
    global_optimizer_step: int,
    total_optimizer_steps: int,
    total_elapsed_sec: float,
    best_val: float,
    optimizer,
    scheduler,
    training_config: dict,
) -> dict:
    return {
        "epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "total_optimizer_steps": total_optimizer_steps,
        "total_elapsed_sec": total_elapsed_sec,
        "best_val": best_val,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_config": training_config,
        "rng_state": capture_rng_state(),
    }


def save_checkpoint_bundle(
    checkpoint_dir: Path,
    model: GPT2LMHeadModel,
    stats: dict,
    trainer_state: dict,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    torch.save(trainer_state, trainer_state_path(checkpoint_dir))


def resolve_paths(args) -> RunPaths:
    return RunPaths(
        tokens=args.tokens_dir.expanduser().resolve(),
        meta=args.meta_path.expanduser().resolve(),
        checkpoints=args.checkpoints_dir.expanduser().resolve(),
        logs=args.logs_dir.expanduser().resolve(),
    )


def ensure_dirs(paths: RunPaths):
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_loaders(tokens_dir: Path, args):
    train_ds = MidiWindowDataset(tokens_dir / "train", args.block_size, args.stride)
    val_ds = MidiWindowDataset(tokens_dir / "validation", args.block_size, args.stride)

    if len(train_ds) == 0:
        raise RuntimeError("no training windows found in the tokenized train split")
    if len(val_ds) == 0:
        raise RuntimeError("no validation windows found in the tokenized validation split")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_ds, val_ds, train_loader, val_loader


def load_model(args, config, expected: dict, training_config: dict) -> tuple[GPT2LMHeadModel, ResumeInfo]:
    checkpoint = resolve_resume_path(args)
    resume = ResumeInfo(checkpoint=checkpoint)

    if checkpoint is None:
        return GPT2LMHeadModel(config), resume

    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")

    stats_path = checkpoint / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"resume stats not found: {stats_path}")

    stats = json.loads(stats_path.read_text())
    resume.start_epoch = int(stats["epoch"]) + 1
    resume.elapsed = float(stats.get("total_elapsed_sec", 0.0))
    resume.best_val = float(stats.get("best_val_so_far", stats.get("val_loss", float("inf"))))

    print(f"loading checkpoint: {checkpoint}", flush=True)
    print(f"resuming from completed epoch: {resume.start_epoch - 1}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    resume.verification = verify_resume_checkpoint(model, checkpoint, expected)
    resume.trainer_state = load_trainer_state(checkpoint)

    if resume.trainer_state is None:
        resume.note = {
            "mode": "model_only",
            "message": (
                "trainer_state.pt not found; restored model weights and "
                "will reconstruct optimizer schedule from the completed epoch"
            ),
        }
    else:
        check_resume_args(resume.trainer_state.get("training_config"), training_config)
        resume.note = {
            "mode": "full_training_state",
            "message": "restored optimizer, scheduler, and RNG state",
        }

    return model, resume


def make_optimizer(model, args):
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def make_scheduler(optimizer, args, total_steps: int, start_step: int):
    if start_step <= 0:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["initial_lr"] = args.learning_rate

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        last_epoch=start_step - 1,
    )


def setup_training_state(
    model,
    args,
    *,
    resume: ResumeInfo,
    optimizer_steps_per_epoch: int,
    total_steps: int,
    device,
):
    optimizer = make_optimizer(model, args)

    if resume.trainer_state is None:
        global_step = (resume.start_epoch - 1) * optimizer_steps_per_epoch
        scheduler = make_scheduler(optimizer, args, total_steps, global_step)
        return optimizer, scheduler, global_step

    optimizer.load_state_dict(resume.trainer_state["optimizer"])
    move_optimizer_state_to_device(optimizer, device)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    scheduler.load_state_dict(resume.trainer_state["scheduler"])

    resume.elapsed = float(resume.trainer_state.get("total_elapsed_sec", resume.elapsed))
    resume.best_val = float(resume.trainer_state.get("best_val", resume.best_val))
    restore_rng_state(resume.trainer_state.get("rng_state"))

    global_step = int(resume.trainer_state["global_optimizer_step"])
    return optimizer, scheduler, global_step


def print_run_header(
    args,
    *,
    device,
    paths: RunPaths,
    vocab_size: int,
    train_ds,
    val_ds,
    train_loader,
    optimizer_steps_per_epoch: int,
    total_steps: int,
    global_step: int,
    resume: ResumeInfo,
):
    print("", flush=True)
    print("===== RUN SUMMARY =====", flush=True)
    print(f"device: {device}", flush=True)
    print(f"tokens_dir: {paths.tokens}", flush=True)
    print(f"vocab_size: {vocab_size}", flush=True)
    print(f"block_size: {args.block_size}", flush=True)
    print(f"stride: {args.stride}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"grad_accum: {args.grad_accum}", flush=True)
    print(f"epochs: {args.epochs}", flush=True)
    print(f"start_epoch: {resume.start_epoch}", flush=True)
    print(f"learning_rate: {args.learning_rate}", flush=True)
    print(f"weight_decay: {args.weight_decay}", flush=True)
    print(f"train_windows: {len(train_ds)}", flush=True)
    print(f"val_windows: {len(val_ds)}", flush=True)
    print(f"train_batches_per_epoch: {len(train_loader)}", flush=True)
    print(f"optimizer_steps_per_epoch: {optimizer_steps_per_epoch}", flush=True)
    print(f"total_optimizer_steps: {total_steps}", flush=True)
    print(f"starting_global_optimizer_step: {global_step}", flush=True)

    if resume.checkpoint is not None:
        print(f"resume_from: {resume.checkpoint}", flush=True)
        print(f"resume_mode: {resume.note['mode']}", flush=True)
        print(f"best_val_so_far: {resume.best_val:.4f}", flush=True)
        if resume.verification["mode"] == "full":
            print(
                "resume_verification: ok "
                f"fingerprint={resume.verification['fingerprint']} "
                f"tensors={resume.verification['tensor_count']} "
                f"params={resume.verification['param_count']}",
                flush=True,
            )
        else:
            print(f"resume_verification: {resume.verification['message']}", flush=True)
        print(f"resume_state: {resume.note['message']}", flush=True)

    print("=======================", flush=True)
    print("", flush=True)


def write_startup(args, status_path: Path, *, device, optimizer_steps_per_epoch: int, total_steps: int, global_step: int, resume: ResumeInfo):
    write_status(
        status_path,
        {
            "phase": "startup",
            "device": str(device),
            "epochs_total": args.epochs,
            "start_epoch": resume.start_epoch,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "global_optimizer_step": global_step,
            "resume_from": str(resume.checkpoint) if resume.checkpoint is not None else None,
            "resume_verification": resume.verification,
            "resume_state": resume.note,
        },
    )


def finish_early(args, status_path: Path, *, resume: ResumeInfo, global_step: int) -> bool:
    if resume.start_epoch <= args.epochs:
        return False

    print(
        f"nothing to do: checkpoint is already at epoch {resume.start_epoch - 1} "
        f"and requested total epochs is {args.epochs}",
        flush=True,
    )
    write_status(
        status_path,
        {
            "phase": "finished",
            "epochs_total": args.epochs,
            "global_optimizer_step": global_step,
            "best_val_so_far": resume.best_val,
            "message": "training already complete for the requested epoch count",
        },
    )
    return True


def accumulation_group(step: int, total_batches: int, grad_accum: int) -> tuple[int, int]:
    group_start = ((step - 1) // grad_accum) * grad_accum + 1
    group_end = min(group_start + grad_accum - 1, total_batches)
    return group_end, group_end - group_start + 1


def progress_note(
    args,
    *,
    epoch: int,
    epoch_step: int,
    optimizer_steps_per_epoch: int,
    global_step: int,
    total_steps: int,
    train_loss: float,
    learning_rate: float,
    elapsed_epoch: float,
    elapsed_total: float,
) -> dict:
    epoch_progress = epoch_step / optimizer_steps_per_epoch
    total_progress = global_step / total_steps
    epoch_bar = fmt_progress_bar(epoch_progress)
    total_bar = fmt_progress_bar(total_progress)
    eta_epoch = (elapsed_epoch / max(epoch_progress, 1e-9)) * (1 - epoch_progress)
    eta_total = (elapsed_total / max(total_progress, 1e-9)) * (1 - total_progress)

    return {
        "epoch_progress": epoch_progress,
        "total_progress": total_progress,
        "epoch_bar": epoch_bar,
        "total_bar": total_bar,
        "eta_epoch": eta_epoch,
        "eta_total": eta_total,
        "line": (
            f"[epoch {epoch}/{args.epochs}] "
            f"ep[{epoch_bar}] {epoch_progress * 100:5.1f}% "
            f"tot[{total_bar}] {total_progress * 100:5.1f}% "
            f"opt_step={epoch_step}/{optimizer_steps_per_epoch} "
            f"global_step={global_step}/{total_steps} "
            f"train_loss={train_loss:.4f} "
            f"lr={learning_rate:.6g} "
            f"elapsed_epoch={fmt_seconds(elapsed_epoch)} "
            f"eta_epoch={fmt_seconds(eta_epoch)} "
            f"elapsed_total={fmt_seconds(elapsed_total)} "
            f"eta_total={fmt_seconds(eta_total)}"
        ),
    }


def log_progress(
    args,
    status_path: Path,
    *,
    epoch: int,
    epoch_step: int,
    optimizer_steps_per_epoch: int,
    global_step: int,
    total_steps: int,
    train_loss: float,
    learning_rate: float,
    elapsed_epoch: float,
    elapsed_total: float,
):
    note = progress_note(
        args,
        epoch=epoch,
        epoch_step=epoch_step,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        global_step=global_step,
        total_steps=total_steps,
        train_loss=train_loss,
        learning_rate=learning_rate,
        elapsed_epoch=elapsed_epoch,
        elapsed_total=elapsed_total,
    )
    print(note["line"], flush=True)
    write_status(
        status_path,
        {
            "phase": "training",
            "epoch": epoch,
            "epochs_total": args.epochs,
            "epoch_optimizer_step": epoch_step,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "global_optimizer_step": global_step,
            "total_optimizer_steps": total_steps,
            "epoch_progress_percent": round(note["epoch_progress"] * 100, 2),
            "epoch_progress_bar": note["epoch_bar"],
            "train_loss_running": train_loss,
            "learning_rate": learning_rate,
            "elapsed_epoch_sec": elapsed_epoch,
            "eta_epoch_sec": note["eta_epoch"],
            "elapsed_total_sec": elapsed_total,
            "eta_total_sec": note["eta_total"],
            "total_progress_percent": round(note["total_progress"] * 100, 2),
            "total_progress_bar": note["total_bar"],
        },
    )


def run_eval(model, val_loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            losses.append(model(input_ids=batch, labels=batch).loss.item())
    return sum(losses) / max(1, len(losses))


def train_epoch(
    args,
    *,
    model,
    optimizer,
    scheduler,
    train_loader,
    device,
    epoch: int,
    run_start: float,
    previous_elapsed: float,
    optimizer_steps_per_epoch: int,
    total_steps: int,
    global_step: int,
    status_path: Path,
) -> tuple[float, float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = 0.0
    seen = 0
    epoch_step = 0
    epoch_start = time.time()
    total_batches = len(train_loader)

    print(f"--- epoch {epoch}/{args.epochs} started ---", flush=True)

    for batch_index, batch in enumerate(train_loader, start=1):
        batch = batch.to(device)
        group_end, scale = accumulation_group(batch_index, total_batches, args.grad_accum)

        out = model(input_ids=batch, labels=batch)
        (out.loss / scale).backward()
        running += out.loss.item()
        seen += 1

        if batch_index != group_end:
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        epoch_step += 1
        global_step += 1

        if epoch_step % args.log_every_opt_steps != 0 and epoch_step != optimizer_steps_per_epoch:
            continue

        elapsed_epoch = time.time() - epoch_start
        elapsed_total = previous_elapsed + (time.time() - run_start)
        log_progress(
            args,
            status_path,
            epoch=epoch,
            epoch_step=epoch_step,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            global_step=global_step,
            total_steps=total_steps,
            train_loss=running / max(1, seen),
            learning_rate=scheduler.get_last_lr()[0],
            elapsed_epoch=elapsed_epoch,
            elapsed_total=elapsed_total,
        )

    return running / max(1, seen), time.time() - epoch_start, global_step


def summarize_epoch(
    args,
    *,
    model,
    val_loader,
    device,
    train_loss: float,
    epoch: int,
    epoch_time: float,
    previous_elapsed: float,
    run_start: float,
    global_step: int,
) -> EpochSummary:
    val_loss = run_eval(model, val_loader, device)
    total_elapsed = previous_elapsed + (time.time() - run_start)
    epochs_left = args.epochs - epoch
    return EpochSummary(
        train_loss=train_loss,
        val_loss=val_loss,
        epoch_time=epoch_time,
        total_elapsed=total_elapsed,
        epochs_left=epochs_left,
        eta_next=epoch_time * epochs_left,
        global_step=global_step,
    )


def build_epoch_files(
    epoch: int,
    summary: EpochSummary,
    *,
    best_val: float,
    total_steps: int,
    training_config: dict,
    optimizer,
    scheduler,
) -> tuple[dict, dict, float]:
    best_after = min(best_val, summary.val_loss)
    stats = {
        "epoch": epoch,
        "train_loss": summary.train_loss,
        "val_loss": summary.val_loss,
        "epoch_time_sec": summary.epoch_time,
        "total_elapsed_sec": summary.total_elapsed,
        "epochs_left": summary.epochs_left,
        "eta_if_same_speed_sec": summary.eta_next,
        "global_optimizer_step": summary.global_step,
        "best_val_so_far": best_after,
        "training_config": training_config,
    }
    trainer_state = build_trainer_state(
        epoch=epoch,
        global_optimizer_step=summary.global_step,
        total_optimizer_steps=total_steps,
        total_elapsed_sec=summary.total_elapsed,
        best_val=best_after,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
    )
    return stats, trainer_state, best_after


def save_epoch(
    *,
    epoch: int,
    summary: EpochSummary,
    best_val: float,
    total_steps: int,
    checkpoints_dir: Path,
    model,
    optimizer,
    scheduler,
    training_config: dict,
) -> float:
    stats, trainer_state, best_after = build_epoch_files(
        epoch,
        summary,
        best_val=best_val,
        total_steps=total_steps,
        training_config=training_config,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    save_checkpoint_bundle(checkpoints_dir / f"epoch_{epoch:02d}", model, stats, trainer_state)

    print(
        f"epoch={epoch}/{epoch + summary.epochs_left} "
        f"train_loss={summary.train_loss:.4f} "
        f"val_loss={summary.val_loss:.4f} "
        f"epoch_time={fmt_seconds(summary.epoch_time)} "
        f"total_elapsed={fmt_seconds(summary.total_elapsed)} "
        f"eta_if_same_speed={fmt_seconds(summary.eta_next)}",
        flush=True,
    )

    if summary.val_loss < best_val:
        best_stats = dict(stats)
        best_stats["best_val_so_far"] = summary.val_loss
        best_state = dict(trainer_state)
        best_state["best_val"] = summary.val_loss
        save_checkpoint_bundle(checkpoints_dir / "best", model, best_stats, best_state)
        print("  -> saved new best checkpoint", flush=True)
        return summary.val_loss

    return best_after


def write_epoch_end(args, status_path: Path, *, epoch: int, summary: EpochSummary, best_val: float):
    write_status(
        status_path,
        {
            "phase": "epoch_end",
            "epoch": epoch,
            "epochs_total": args.epochs,
            "train_loss": summary.train_loss,
            "val_loss": summary.val_loss,
            "epoch_time_sec": summary.epoch_time,
            "total_elapsed_sec": summary.total_elapsed,
            "epochs_left": summary.epochs_left,
            "eta_if_same_speed_sec": summary.eta_next,
            "best_val_so_far": best_val,
            "global_optimizer_step": summary.global_step,
        },
    )


def write_finished(status_path: Path, *, total_time: float, global_step: int, best_val: float):
    write_status(
        status_path,
        {
            "phase": "finished",
            "total_time_sec": total_time,
            "global_optimizer_step": global_step,
            "best_val_so_far": best_val,
        },
    )


def write_interrupted(status_path: Path, *, epoch: int | None, global_step: int, total_elapsed: float):
    write_status(
        status_path,
        {
            "phase": "interrupted",
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "total_elapsed_sec": total_elapsed,
        },
    )


def main():
    args = parse_args()
    validate_args(args)

    paths = resolve_paths(args)
    ensure_dirs(paths)

    set_random_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = pick_device()
    print(f"device: {device}", flush=True)

    vocab_size = load_vocab_size(paths.meta)
    expected = model_shape(vocab_size, args.block_size)
    training_config = run_config(args, vocab_size)

    train_ds, val_ds, train_loader, val_loader = build_loaders(paths.tokens, args)
    model_config = build_model_config(vocab_size, args.block_size)
    model, resume = load_model(args, model_config, expected, training_config)
    model.to(device)

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    optimizer, scheduler, global_step = setup_training_state(
        model,
        args,
        resume=resume,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_steps=total_steps,
        device=device,
    )

    print_run_header(
        args,
        device=device,
        paths=paths,
        vocab_size=vocab_size,
        train_ds=train_ds,
        val_ds=val_ds,
        train_loader=train_loader,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_steps=total_steps,
        global_step=global_step,
        resume=resume,
    )
    write_startup(
        args,
        paths.status,
        device=device,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_steps=total_steps,
        global_step=global_step,
        resume=resume,
    )

    if finish_early(args, paths.status, resume=resume, global_step=global_step):
        return

    run_start = time.time()

    try:
        for epoch in range(resume.start_epoch, args.epochs + 1):
            train_loss, epoch_time, global_step = train_epoch(
                args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
                run_start=run_start,
                previous_elapsed=resume.elapsed,
                optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                total_steps=total_steps,
                global_step=global_step,
                status_path=paths.status,
            )
            summary = summarize_epoch(
                args,
                model=model,
                val_loader=val_loader,
                device=device,
                train_loss=train_loss,
                epoch=epoch,
                epoch_time=epoch_time,
                previous_elapsed=resume.elapsed,
                run_start=run_start,
                global_step=global_step,
            )
            resume.best_val = save_epoch(
                epoch=epoch,
                summary=summary,
                best_val=resume.best_val,
                total_steps=total_steps,
                checkpoints_dir=paths.checkpoints,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=training_config,
            )
            write_epoch_end(
                args,
                paths.status,
                epoch=epoch,
                summary=summary,
                best_val=resume.best_val,
            )

        print("", flush=True)
        print("training complete", flush=True)
        total_time = resume.elapsed + (time.time() - run_start)
        print(f"total_time={fmt_seconds(total_time)}", flush=True)
        write_finished(
            paths.status,
            total_time=total_time,
            global_step=global_step,
            best_val=resume.best_val,
        )

    except KeyboardInterrupt:
        print("", flush=True)
        print("training interrupted by user", flush=True)
        total_time = resume.elapsed + (time.time() - run_start)
        write_interrupted(
            paths.status,
            epoch=epoch if "epoch" in locals() else None,
            global_step=global_step,
            total_elapsed=total_time,
        )
        raise


if __name__ == "__main__":
    main()
