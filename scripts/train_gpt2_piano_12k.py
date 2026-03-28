from pathlib import Path
import hashlib
import json
import math
import time
import random
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from _paths import ROOT

ART = ROOT / "artifacts"
TOKENS = ART / "tokens"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

META = json.loads((ART / "meta.json").read_text())
VOCAB_SIZE = int(META["vocab_size"])

# ===== TRAINING CONFIG =====
BLOCK_SIZE = 2048
STRIDE = 1024

BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 8
LR = 2e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
SEED = 42
RESUME_FROM = CHECKPOINTS / "epoch_01"  # set to None to train from scratch

LOG_EVERY_OPT_STEPS = 25

MODEL_CFG = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "n_positions": BLOCK_SIZE,
    "n_ctx": BLOCK_SIZE,
}
# ==========================

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}", flush=True)

STATUS_PATH = LOGS / "run_status.json"


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def write_status(payload: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def fmt_progress_bar(progress: float, width: int = 16) -> str:
    progress = min(max(progress, 0.0), 1.0)

    if progress >= 1.0:
        return "#" * width

    filled = int(progress * width)
    if filled <= 0:
        return ">" + "." * (width - 1)

    return "#" * (filled - 1) + ">" + "." * (width - filled)


def expected_model_config() -> dict:
    return {
        "vocab_size": VOCAB_SIZE,
        "n_positions": MODEL_CFG["n_positions"],
        "n_ctx": MODEL_CFG["n_ctx"],
        "n_embd": MODEL_CFG["n_embd"],
        "n_layer": MODEL_CFG["n_layer"],
        "n_head": MODEL_CFG["n_head"],
    }


def snapshot_model_config(config_or_dict) -> dict:
    if isinstance(config_or_dict, dict):
        source = config_or_dict
    else:
        source = config_or_dict.to_dict()

    return {key: source[key] for key in expected_model_config()}


def assert_expected_model_config(config_or_dict, label: str):
    actual = snapshot_model_config(config_or_dict)
    expected = expected_model_config()

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

    if mismatches:
        raise RuntimeError(
            f"{label} config mismatch:\n" + "\n".join(mismatches)
        )


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


def verify_resume_checkpoint(model: GPT2LMHeadModel, resume_dir: Path) -> dict:
    assert_expected_model_config(model.config, "loaded model")
    assert_expected_model_config(
        json.loads((resume_dir / "config.json").read_text()),
        f"{resume_dir / 'config.json'}",
    )

    model_path = resume_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"resume model weights not found: {model_path}")

    try:
        from safetensors.torch import load_file
    except ImportError:
        return {
            "mode": "config_only",
            "message": "safetensors unavailable; verified config only",
        }

    checkpoint_state = load_file(str(model_path))
    model_state = model.state_dict()

    missing_keys = sorted(set(checkpoint_state) - set(model_state))
    if missing_keys:
        raise RuntimeError(
            f"loaded model is missing checkpoint tensor: {missing_keys[0]}"
        )

    for name in sorted(checkpoint_state):
        expected = checkpoint_state[name]
        actual = model_state[name].detach().cpu()

        if expected.shape != actual.shape:
            raise RuntimeError(
                f"resume tensor shape mismatch for {name}: "
                f"expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
            )

        if expected.dtype != actual.dtype:
            raise RuntimeError(
                f"resume tensor dtype mismatch for {name}: "
                f"expected {expected.dtype}, got {actual.dtype}"
            )

        if not torch.equal(expected, actual):
            max_abs_diff = (expected.to(torch.float32) - actual.to(torch.float32)).abs().max().item()
            raise RuntimeError(
                f"resume tensor value mismatch for {name}: max_abs_diff={max_abs_diff}"
            )

    fingerprint, tensor_count, param_count = state_dict_fingerprint(checkpoint_state)
    return {
        "mode": "full",
        "fingerprint": fingerprint,
        "tensor_count": tensor_count,
        "param_count": param_count,
    }


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

            for s in starts:
                self.index.append((arr_id, s))

        print(
            f"{split_dir.name}: files={len(self.arrays)} windows={len(self.index)}",
            flush=True
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        arr_id, start = self.index[idx]
        arr = self.arrays[arr_id]
        x = np.asarray(arr[start:start + self.block_size], dtype=np.int64)
        return torch.tensor(x, dtype=torch.long)


train_ds = MidiWindowDataset(TOKENS / "train", BLOCK_SIZE, STRIDE)
val_ds = MidiWindowDataset(TOKENS / "validation", BLOCK_SIZE, STRIDE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=MODEL_CFG["n_positions"],
    n_ctx=MODEL_CFG["n_ctx"],
    n_embd=MODEL_CFG["n_embd"],
    n_layer=MODEL_CFG["n_layer"],
    n_head=MODEL_CFG["n_head"],
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    use_cache=False,
)

resume_stats = None
resume_verification = None
start_epoch = 1
global_opt_step = 0
previous_total_elapsed = 0.0
best_val = float("inf")

if RESUME_FROM is None:
    model = GPT2LMHeadModel(config)
else:
    if not RESUME_FROM.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {RESUME_FROM}")

    stats_path = RESUME_FROM / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"resume stats not found: {stats_path}")

    resume_stats = json.loads(stats_path.read_text())
    completed_epoch = int(resume_stats["epoch"])
    start_epoch = completed_epoch + 1
    previous_total_elapsed = float(resume_stats.get("total_elapsed_sec", 0.0))
    best_val = float(resume_stats.get("val_loss", float("inf")))

    print(f"loading checkpoint: {RESUME_FROM}", flush=True)
    print(f"resuming from completed epoch: {completed_epoch}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(RESUME_FROM)
    resume_verification = verify_resume_checkpoint(model, RESUME_FROM)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
total_optimizer_steps = optimizer_steps_per_epoch * EPOCHS
global_opt_step = (start_epoch - 1) * optimizer_steps_per_epoch

for group in optimizer.param_groups:
    group.setdefault("initial_lr", group["lr"])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_optimizer_steps,
    last_epoch=global_opt_step - 1,
)

print("", flush=True)
print("===== RUN SUMMARY =====", flush=True)
print(f"device: {device}", flush=True)
print(f"vocab_size: {VOCAB_SIZE}", flush=True)
print(f"block_size: {BLOCK_SIZE}", flush=True)
print(f"stride: {STRIDE}", flush=True)
print(f"batch_size: {BATCH_SIZE}", flush=True)
print(f"grad_accum: {GRAD_ACCUM}", flush=True)
print(f"epochs: {EPOCHS}", flush=True)
print(f"start_epoch: {start_epoch}", flush=True)
print(f"learning_rate: {LR}", flush=True)
print(f"weight_decay: {WEIGHT_DECAY}", flush=True)
print(f"train_windows: {len(train_ds)}", flush=True)
print(f"val_windows: {len(val_ds)}", flush=True)
print(f"train_batches_per_epoch: {len(train_loader)}", flush=True)
print(f"optimizer_steps_per_epoch: {optimizer_steps_per_epoch}", flush=True)
print(f"total_optimizer_steps: {total_optimizer_steps}", flush=True)
print(f"starting_global_optimizer_step: {global_opt_step}", flush=True)
if RESUME_FROM is not None:
    print(f"resume_from: {RESUME_FROM}", flush=True)
    print(f"best_val_so_far: {best_val:.4f}", flush=True)
    if resume_verification["mode"] == "full":
        print(
            "resume_verification: ok "
            f"fingerprint={resume_verification['fingerprint']} "
            f"tensors={resume_verification['tensor_count']} "
            f"params={resume_verification['param_count']}",
            flush=True,
        )
    else:
        print(
            f"resume_verification: {resume_verification['message']}",
            flush=True,
        )
print("=======================", flush=True)
print("", flush=True)

write_status({
    "phase": "startup",
    "device": str(device),
    "epochs_total": EPOCHS,
    "start_epoch": start_epoch,
    "block_size": BLOCK_SIZE,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    "total_optimizer_steps": total_optimizer_steps,
    "global_optimizer_step": global_opt_step,
    "resume_from": str(RESUME_FROM) if RESUME_FROM is not None else None,
    "resume_verification": resume_verification,
})


def run_eval():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(input_ids=batch, labels=batch)
            losses.append(out.loss.item())
    return sum(losses) / max(1, len(losses))


run_start = time.time()

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        count = 0
        epoch_start = time.time()
        epoch_opt_steps = 0

        print(f"--- epoch {epoch}/{EPOCHS} started ---", flush=True)

        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)

            # Determine the actual size of the current accumulation group.
            group_start = ((step - 1) // GRAD_ACCUM) * GRAD_ACCUM + 1
            group_end = min(group_start + GRAD_ACCUM - 1, num_batches)
            current_accum = group_end - group_start + 1

            out = model(input_ids=batch, labels=batch)
            loss = out.loss / current_accum
            loss.backward()

            running += out.loss.item()
            count += 1

            if step == group_end:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_opt_steps += 1
                global_opt_step += 1

                if epoch_opt_steps % LOG_EVERY_OPT_STEPS == 0 or epoch_opt_steps == optimizer_steps_per_epoch:
                    elapsed_epoch = time.time() - epoch_start
                    elapsed_total = previous_total_elapsed + (time.time() - run_start)

                    epoch_progress = epoch_opt_steps / optimizer_steps_per_epoch
                    total_progress = global_opt_step / total_optimizer_steps
                    epoch_bar = fmt_progress_bar(epoch_progress)
                    total_bar = fmt_progress_bar(total_progress)

                    eta_epoch = (elapsed_epoch / max(epoch_progress, 1e-9)) * (1 - epoch_progress)
                    eta_total = (elapsed_total / max(total_progress, 1e-9)) * (1 - total_progress)

                    avg_train_loss = running / max(1, count)
                    current_lr = scheduler.get_last_lr()[0]

                    line = (
                        f"[epoch {epoch}/{EPOCHS}] "
                        f"ep[{epoch_bar}] {epoch_progress*100:5.1f}% "
                        f"tot[{total_bar}] {total_progress*100:5.1f}% "
                        f"opt_step={epoch_opt_steps}/{optimizer_steps_per_epoch} "
                        f"global_step={global_opt_step}/{total_optimizer_steps} "
                        f"train_loss={avg_train_loss:.4f} "
                        f"lr={current_lr:.6g} "
                        f"elapsed_epoch={fmt_seconds(elapsed_epoch)} "
                        f"eta_epoch={fmt_seconds(eta_epoch)} "
                        f"elapsed_total={fmt_seconds(elapsed_total)} "
                        f"eta_total={fmt_seconds(eta_total)}"
                    )
                    print(line, flush=True)

                    write_status({
                        "phase": "training",
                        "epoch": epoch,
                        "epochs_total": EPOCHS,
                        "epoch_optimizer_step": epoch_opt_steps,
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "global_optimizer_step": global_opt_step,
                        "total_optimizer_steps": total_optimizer_steps,
                        "epoch_progress_percent": round(epoch_progress * 100, 2),
                        "epoch_progress_bar": epoch_bar,
                        "train_loss_running": avg_train_loss,
                        "learning_rate": current_lr,
                        "elapsed_epoch_sec": elapsed_epoch,
                        "eta_epoch_sec": eta_epoch,
                        "elapsed_total_sec": elapsed_total,
                        "eta_total_sec": eta_total,
                        "total_progress_percent": round(total_progress * 100, 2),
                        "total_progress_bar": total_bar,
                    })

        val_loss = run_eval()
        train_loss = running / max(1, count)
        epoch_time = time.time() - epoch_start
        total_elapsed = previous_total_elapsed + (time.time() - run_start)
        epochs_left = EPOCHS - epoch
        eta_from_last_epoch = epoch_time * epochs_left

        ckpt_dir = CHECKPOINTS / f"epoch_{epoch:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)

        stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time,
            "total_elapsed_sec": total_elapsed,
            "epochs_left": epochs_left,
            "eta_if_same_speed_sec": eta_from_last_epoch,
        }
        (ckpt_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        print(
            f"epoch={epoch}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"epoch_time={fmt_seconds(epoch_time)} "
            f"total_elapsed={fmt_seconds(total_elapsed)} "
            f"eta_if_same_speed={fmt_seconds(eta_from_last_epoch)}",
            flush=True
        )

        if val_loss < best_val:
            best_val = val_loss
            best_dir = CHECKPOINTS / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            (best_dir / "stats.json").write_text(json.dumps(stats, indent=2))
            print("  -> saved new best checkpoint", flush=True)

        write_status({
            "phase": "epoch_end",
            "epoch": epoch,
            "epochs_total": EPOCHS,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time,
            "total_elapsed_sec": total_elapsed,
            "epochs_left": epochs_left,
            "eta_if_same_speed_sec": eta_from_last_epoch,
            "best_val_so_far": best_val,
        })

    print("", flush=True)
    print("training complete", flush=True)
    print(
        f"total_time={fmt_seconds(previous_total_elapsed + (time.time() - run_start))}",
        flush=True,
    )

    write_status({
        "phase": "finished",
        "total_time_sec": previous_total_elapsed + (time.time() - run_start),
    })

except KeyboardInterrupt:
    print("", flush=True)
    print("training interrupted by user", flush=True)
    write_status({
        "phase": "interrupted",
        "epoch": epoch if 'epoch' in locals() else None,
        "global_optimizer_step": global_opt_step,
        "total_elapsed_sec": previous_total_elapsed + (time.time() - run_start),
    })
    raise
