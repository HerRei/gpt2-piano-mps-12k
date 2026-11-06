"""Train the v1 GPT-2 on the REMI tokens written by prepare_data.py.

usage: python v1/train.py --data data/v1 --out runs/v1 [--epochs 4] [--bf16]

Same recipe as the original run on an M1 Max: 12 layers x 768, 2,048-token windows with stride 1,024,
16 windows per update (batch 2 x 8 accumulation there; --batch 8 --accum 2 gives the same updates faster on a
bigger GPU), AdamW 2e-4 (weight decay 0.01), cosine schedule with 1,000 warmup steps. After each
epoch the validation loss decides <out>/best. <out>/last holds a resumable state, saved every --save-minutes
and on SIGTERM, and the run continues from it when started again.
"""
import argparse
import json
import math
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup


class Windows(torch.utils.data.Dataset):
    def __init__(self, split_dir, block, stride):
        self.block, self.arrays, self.index = block, [], []
        for path in sorted(Path(split_dir).glob("*.npy")):
            arr = np.load(path).astype(np.int16)  # in RAM: ~160M tokens, and one memmap per file runs out of handles
            if len(arr) < block + 1:
                continue
            last = len(arr) - (block + 1)
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            self.index += [(len(self.arrays), s) for s in starts]
            self.arrays.append(arr)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        a, s = self.index[i]
        return torch.from_numpy(np.asarray(self.arrays[a][s:s + self.block], dtype=np.int64))


def gpt2_config(vocab_size, block):
    return GPT2Config(vocab_size=vocab_size, n_positions=block, n_ctx=block, n_embd=768, n_layer=12, n_head=12,
                      resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, use_cache=False, bos_token_id=None,
                      eos_token_id=None)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save(model, state, dst):
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    if state is not None:
        torch.save(state, tmp / "trainer_state.pt")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA/ROCm); the original run was fp32")
    ap.add_argument("--save-minutes", type=float, default=20)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = device()
    vocab = json.loads((a.data / "meta.json").read_text())["vocab_size"]
    train_ds = Windows(a.data / "train", a.block, a.stride)
    val_ds = Windows(a.data / "validation", a.block, a.stride)
    print(f"train windows {len(train_ds)}, validation windows {len(val_ds)}, vocab {vocab}, device {dev}", flush=True)

    batches = math.ceil(len(train_ds) / a.batch)
    steps_per_epoch = math.ceil(batches / a.accum)
    total_steps = steps_per_epoch * a.epochs

    last = a.out / "last"
    if (last / "trainer_state.pt").exists():
        model = GPT2LMHeadModel.from_pretrained(last).to(dev)
        st = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
    else:
        model = GPT2LMHeadModel(gpt2_config(vocab, a.block)).to(dev)
        st = None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total_steps)
    epoch, batch_i, best, history = 0, 0, float("inf"), []
    if st is not None:
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        epoch, batch_i, best, history = st["epoch"], st["batch"], st["best"], st["history"]
        torch.set_rng_state(st["rng"])
        print(f"resumed at epoch {epoch + 1}, batch {batch_i}/{batches}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "log.jsonl", "a")
    if st is None:
        log.write(json.dumps({"event": "start", "total_steps": total_steps,
                              "tok_per_step": a.batch * a.accum * a.block, "time": time.time()}) + "\n")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16 and dev.type == "cuda" else torch.autocast(
        dev.type, enabled=False)

    def state(ep, bi):
        return {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": ep, "batch": bi,
                "best": best, "history": history, "rng": torch.get_rng_state()}

    while epoch < a.epochs:
        order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(a.seed + epoch))
        model.train()
        t_save, t_log, run_loss, run_n = time.time(), time.time(), 0.0, 0
        while batch_i < batches:
            group = range(batch_i, min(batch_i + a.accum, batches))
            for b in group:
                idx = order[b * a.batch:(b + 1) * a.batch]
                x = torch.stack([train_ds[int(i)] for i in idx]).to(dev)
                with amp:
                    loss = model(input_ids=x, labels=x).loss
                (loss / len(group)).backward()
                run_loss, run_n = run_loss + loss.item(), run_n + 1
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            batch_i = group.stop
            step = epoch * steps_per_epoch + math.ceil(batch_i / a.accum)
            if step % 25 == 0:
                now = time.time()
                log.write(json.dumps({"step": step, "epoch": epoch + 1, "loss": round(run_loss / run_n, 4),
                                      "lr": sched.get_last_lr()[0], "tokens": step * a.batch * a.accum * a.block,
                                      "tok_s": round(25 * a.batch * a.accum * a.block / (now - t_log)),
                                      "time": now}) + "\n")
                log.flush()
                t_log, run_loss, run_n = now, 0.0, 0
            if stop or time.time() - t_save > a.save_minutes * 60:
                save(model, state(epoch, batch_i), last)
                t_save = time.time()
                if stop:
                    print(f"stopped at epoch {epoch + 1}, batch {batch_i}; state saved", flush=True)
                    return

        model.eval()
        losses = []
        with torch.no_grad(), amp:
            for i in range(0, len(val_ds), a.batch):
                x = torch.stack([val_ds[j] for j in range(i, min(i + a.batch, len(val_ds)))]).to(dev)
                losses.append(model(input_ids=x, labels=x).loss.item())
        val = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "val_loss": round(val, 4)})
        print(f"epoch {epoch + 1}: validation loss {val:.4f}", flush=True)
        log.write(json.dumps({"epoch": epoch + 1, "val": round(val, 4), "time": time.time()}) + "\n")
        if val < best:
            best = val
            save(model, None, a.out / "best")
        epoch, batch_i = epoch + 1, 0
        save(model, state(epoch, batch_i), last)

    (a.out / "history.json").write_text(json.dumps(history, indent=2))
    log.write(json.dumps({"event": "done", "step": total_steps, "best_val": round(best, 4), "time": time.time()}) + "\n")
    print(f"done; best validation loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
