import argparse
import json
import math
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
EXAMPLES = ARTIFACTS / "examples"
META_PATH = ARTIFACTS / "meta.json"
CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints"
LOGS = EXPERIMENT_ROOT / "logs"
STATUS_PATH = LOGS / "run_status.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the melody intensity editor model."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-opt-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    return parser.parse_args()


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def fmt_progress_bar(progress: float, width: int = 16) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    if filled >= width:
        return "#" * width
    return "#" * filled + ">" + "." * (width - filled - 1)


def write_status(payload: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested mps but MPS is not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PackedSequenceDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.paths = sorted(split_dir.glob("*.npz"))
        print(f"{split_dir.name}: examples={len(self.paths)}", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        payload = np.load(self.paths[idx])
        ids = payload["ids"].astype(np.int64)
        loss_start = int(payload["loss_start"])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "loss_start": loss_start,
        }


def build_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(item["ids"].shape[0] for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for row, item in enumerate(batch):
            ids = item["ids"]
            seq_len = ids.shape[0]
            input_ids[row, :seq_len] = ids
            attention_mask[row, :seq_len] = 1
            labels[row, :seq_len] = ids
            labels[row, :item["loss_start"]] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def build_model_config(meta: dict, args) -> GPT2Config:
    return GPT2Config(
        vocab_size=int(meta["vocab_size"]),
        n_positions=int(meta["max_seq_len"]),
        n_ctx=int(meta["max_seq_len"]),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
        pad_token_id=int(meta["special_tokens"]["PAD"]),
        eos_token_id=int(meta["special_tokens"]["TGT_END"]),
    )


def main():
    args = parse_args()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if not META_PATH.exists():
        raise FileNotFoundError(f"dataset metadata not found: {META_PATH}")

    meta = json.loads(META_PATH.read_text())
    pad_token_id = int(meta["special_tokens"]["PAD"])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    print(f"device: {device}", flush=True)

    train_ds = PackedSequenceDataset(EXAMPLES / "train")
    val_ds = PackedSequenceDataset(EXAMPLES / "validation")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("train/validation example sets are empty")

    collate_fn = build_collate_fn(pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    config = build_model_config(meta, args)

    resume_stats = None
    start_epoch = 1
    previous_total_elapsed = 0.0
    best_val = float("inf")

    if args.resume_from is None:
        model = GPT2LMHeadModel(config)
    else:
        resume_from = args.resume_from.expanduser().resolve()
        stats_path = resume_from / "stats.json"
        if not resume_from.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        if not stats_path.exists():
            raise FileNotFoundError(f"resume stats not found: {stats_path}")
        resume_stats = json.loads(stats_path.read_text())
        start_epoch = int(resume_stats["epoch"]) + 1
        previous_total_elapsed = float(resume_stats.get("total_elapsed_sec", 0.0))
        best_val = float(resume_stats.get("val_loss", float("inf")))
        model = GPT2LMHeadModel.from_pretrained(resume_from, local_files_only=True)

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(device)
    param_count = sum(param.numel() for param in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    global_opt_step = (start_epoch - 1) * optimizer_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_optimizer_steps,
        last_epoch=global_opt_step - 1,
    )

    print("", flush=True)
    print("===== RUN SUMMARY =====", flush=True)
    print(f"device: {device}", flush=True)
    print(f"params: {param_count}", flush=True)
    print(f"vocab_size: {meta['vocab_size']}", flush=True)
    print(f"max_seq_len: {meta['max_seq_len']}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"grad_accum: {args.grad_accum}", flush=True)
    print(f"epochs: {args.epochs}", flush=True)
    print(f"start_epoch: {start_epoch}", flush=True)
    print(f"learning_rate: {args.lr}", flush=True)
    print(f"weight_decay: {args.weight_decay}", flush=True)
    print(f"gradient_checkpointing: {not args.disable_gradient_checkpointing}", flush=True)
    print(f"train_examples: {len(train_ds)}", flush=True)
    print(f"val_examples: {len(val_ds)}", flush=True)
    print(f"train_batches_per_epoch: {len(train_loader)}", flush=True)
    print(f"optimizer_steps_per_epoch: {optimizer_steps_per_epoch}", flush=True)
    print(f"total_optimizer_steps: {total_optimizer_steps}", flush=True)
    print("=======================", flush=True)
    print("", flush=True)

    write_status(
        {
            "phase": "startup",
            "device": str(device),
            "params": param_count,
            "epochs_total": args.epochs,
            "start_epoch": start_epoch,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "global_optimizer_step": global_opt_step,
            "resume_from": str(args.resume_from.expanduser().resolve()) if args.resume_from is not None else None,
        }
    )

    def run_eval():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                losses.append(outputs.loss.item())
        return sum(losses) / max(1, len(losses))

    run_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            running = 0.0
            count = 0
            epoch_start = time.time()
            epoch_opt_steps = 0
            num_batches = len(train_loader)

            print(f"--- epoch {epoch}/{args.epochs} started ---", flush=True)

            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}

                group_start = ((step - 1) // args.grad_accum) * args.grad_accum + 1
                group_end = min(group_start + args.grad_accum - 1, num_batches)
                current_accum = group_end - group_start + 1

                outputs = model(**batch)
                loss = outputs.loss / current_accum
                loss.backward()

                running += outputs.loss.item()
                count += 1

                if step == group_end:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    epoch_opt_steps += 1
                    global_opt_step += 1

                    if epoch_opt_steps % args.log_every_opt_steps == 0 or epoch_opt_steps == optimizer_steps_per_epoch:
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
                            f"[epoch {epoch}/{args.epochs}] "
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

                        write_status(
                            {
                                "phase": "training",
                                "epoch": epoch,
                                "epochs_total": args.epochs,
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
                            }
                        )

            val_loss = run_eval()
            train_loss = running / max(1, count)
            epoch_time = time.time() - epoch_start
            total_elapsed = previous_total_elapsed + (time.time() - run_start)
            epochs_left = args.epochs - epoch
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
                "params": param_count,
                "task": "melody_intensity_editor",
            }
            (ckpt_dir / "stats.json").write_text(json.dumps(stats, indent=2))

            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"epoch_time={fmt_seconds(epoch_time)} "
                f"total_elapsed={fmt_seconds(total_elapsed)} "
                f"eta_if_same_speed={fmt_seconds(eta_from_last_epoch)}",
                flush=True,
            )

            if val_loss < best_val:
                best_val = val_loss
                best_dir = CHECKPOINTS / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                (best_dir / "stats.json").write_text(json.dumps(stats, indent=2))
                print("  -> saved new best checkpoint", flush=True)

            write_status(
                {
                    "phase": "epoch_end",
                    "epoch": epoch,
                    "epochs_total": args.epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time_sec": epoch_time,
                    "total_elapsed_sec": total_elapsed,
                    "epochs_left": epochs_left,
                    "eta_if_same_speed_sec": eta_from_last_epoch,
                    "best_val_so_far": best_val,
                }
            )

        print("", flush=True)
        print("training complete", flush=True)
        print(f"total_time={fmt_seconds(previous_total_elapsed + (time.time() - run_start))}", flush=True)
        write_status({"phase": "finished", "total_time_sec": previous_total_elapsed + (time.time() - run_start)})
    except KeyboardInterrupt:
        print("", flush=True)
        print("training interrupted by user", flush=True)
        write_status(
            {
                "phase": "interrupted",
                "epoch": epoch if "epoch" in locals() else None,
                "global_optimizer_step": global_opt_step,
                "total_elapsed_sec": previous_total_elapsed + (time.time() - run_start),
            }
        )
        raise


if __name__ == "__main__":
    main()
