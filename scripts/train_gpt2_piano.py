from pathlib import Path
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

BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
SEED = 42

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

        print(f"{split_dir.name}: files={len(self.arrays)} windows={len(self.index)}", flush=True)

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

model = GPT2LMHeadModel(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
total_optimizer_steps = optimizer_steps_per_epoch * EPOCHS

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_optimizer_steps,
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
print(f"learning_rate: {LR}", flush=True)
print(f"weight_decay: {WEIGHT_DECAY}", flush=True)
print(f"train_windows: {len(train_ds)}", flush=True)
print(f"val_windows: {len(val_ds)}", flush=True)
print(f"train_batches_per_epoch: {len(train_loader)}", flush=True)
print(f"optimizer_steps_per_epoch: {optimizer_steps_per_epoch}", flush=True)
print(f"total_optimizer_steps: {total_optimizer_steps}", flush=True)
print("=======================", flush=True)
print("", flush=True)

write_status({
    "phase": "startup",
    "device": str(device),
    "epochs_total": EPOCHS,
    "block_size": BLOCK_SIZE,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    "total_optimizer_steps": total_optimizer_steps,
})

best_val = float("inf")

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
global_opt_step = 0

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        count = 0
        epoch_start = time.time()
        epoch_opt_steps = 0

        print(f"--- epoch {epoch}/{EPOCHS} started ---", flush=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)

            out = model(input_ids=batch, labels=batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            running += out.loss.item()
            count += 1

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_opt_steps += 1
                global_opt_step += 1

                if epoch_opt_steps % LOG_EVERY_OPT_STEPS == 0 or epoch_opt_steps == optimizer_steps_per_epoch:
                    elapsed_epoch = time.time() - epoch_start
                    elapsed_total = time.time() - run_start

                    epoch_progress = epoch_opt_steps / optimizer_steps_per_epoch
                    total_progress = global_opt_step / total_optimizer_steps

                    eta_epoch = (elapsed_epoch / max(epoch_progress, 1e-9)) * (1 - epoch_progress)
                    eta_total = (elapsed_total / max(total_progress, 1e-9)) * (1 - total_progress)

                    avg_train_loss = running / max(1, count)
                    current_lr = scheduler.get_last_lr()[0]

                    print(
                        f"[epoch {epoch}/{EPOCHS}] "
                        f"opt_step={epoch_opt_steps}/{optimizer_steps_per_epoch} "
                        f"global_step={global_opt_step}/{total_optimizer_steps} "
                        f"epoch_progress={epoch_progress*100:.1f}% "
                        f"train_loss={avg_train_loss:.4f} "
                        f"lr={current_lr:.6g} "
                        f"elapsed_epoch={fmt_seconds(elapsed_epoch)} "
                        f"eta_epoch={fmt_seconds(eta_epoch)} "
                        f"elapsed_total={fmt_seconds(elapsed_total)} "
                        f"eta_total={fmt_seconds(eta_total)}",
                        flush=True
                    )

                    write_status({
                        "phase": "training",
                        "epoch": epoch,
                        "epochs_total": EPOCHS,
                        "epoch_optimizer_step": epoch_opt_steps,
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "global_optimizer_step": global_opt_step,
                        "total_optimizer_steps": total_optimizer_steps,
                        "epoch_progress_percent": round(epoch_progress * 100, 2),
                        "train_loss_running": avg_train_loss,
                        "learning_rate": current_lr,
                        "elapsed_epoch_sec": elapsed_epoch,
                        "eta_epoch_sec": eta_epoch,
                        "elapsed_total_sec": elapsed_total,
                        "eta_total_sec": eta_total,
                    })

        val_loss = run_eval()
        train_loss = running / max(1, count)
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - run_start
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
    print(f"total_time={fmt_seconds(time.time() - run_start)}", flush=True)

    write_status({
        "phase": "finished",
        "total_time_sec": time.time() - run_start,
    })

except KeyboardInterrupt:
    print("", flush=True)
    print("training interrupted by user", flush=True)
    write_status({
        "phase": "interrupted",
        "epoch": epoch if 'epoch' in locals() else None,
        "global_optimizer_step": global_opt_step,
        "total_elapsed_sec": time.time() - run_start,
    })
    raise
