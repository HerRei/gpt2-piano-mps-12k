"""Train the v2 RWKV-7 on Aria-MIDI tokens from tokenize_aria.py, with density and loudness control tokens.

usage (one GPU):
  python v2/train.py --data data/aria --out runs/v2 --tokens 4.59e9
usage (two GPUs, one process each, gradients summed over gloo; the published model used an RX 9060 XT in bf16
and a Radeon VII in fp16 with 20 and 12 micro-batches per step):
  RANK=1 HIP_VISIBLE_DEVICES=1 python v2/train.py ... --world 2 --accum_by_rank 20,12 --dtype_by_rank bfloat16,float16 &
  RANK=0 HIP_VISIBLE_DEVICES=0 python v2/train.py ... --world 2 --accum_by_rank 20,12 --dtype_by_rank bfloat16,float16

Windows of T+1 tokens are drawn inside single files, weighted by length; train and validation are split by
recording. 15 % of the 5-s segments get "none" controls, so the same model can be run without controls and with
classifier-free guidance. Learning rate: warmup, constant, then 1 - sqrt decay over the last --decay_frac.
SIGTERM or SIGINT: finish the step, save <out>/ckpt.pt, exit. Starting again resumes from it (with a new data
order: the sampler is seeded from the clock).
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RWKV7
from vocab import Vocab, aria_tables, levels

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True, help="tokenize_aria.py output directory")
ap.add_argument("--out", required=True)
ap.add_argument("--vocab_map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_map.npy"))
ap.add_argument("--d", type=int, default=768)
ap.add_argument("--L", type=int, default=12)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--tokens", type=float, default=4.59e9, help="total training tokens")
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=500, help="optimizer steps")
ap.add_argument("--decay_frac", type=float, default=0.2)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--beta2", type=float, default=0.99)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--ctrl_dropout", type=float, default=0.15)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--eval_windows", type=int, default=256)
ap.add_argument("--ckpt_minutes", type=float, default=15)
ap.add_argument("--milestone_every", type=int, default=2000, help="keep ckpt_stepN.pt every N steps and ckpt_predecay.pt")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_pct", type=int, default=1)
ap.add_argument("--init_from", default="", help="start a new run from this checkpoint (weights and optimizer)")
ap.add_argument("--reset_opt", type=int, default=0, help="with --init_from: weights only")
ap.add_argument("--eval_only", type=int, default=0, help="print the validation loss per token kind and exit")
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
ap.add_argument("--accum_by_rank", default="", help="micro-batches per optimizer step on each rank, e.g. 20,12")
ap.add_argument("--dtype_by_rank", default="", help="e.g. bfloat16,float16")
ap.add_argument("--dist_init", default="tcp://127.0.0.1:29541")
ap.add_argument("--resync_every", type=int, default=200, help="broadcast rank 0's weights every N steps")
args = ap.parse_args()

DIST = args.world > 1
if DIST:
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=args.dist_init, rank=args.rank, world_size=args.world)
    accs = [int(a) for a in args.accum_by_rank.split(",")] if args.accum_by_rank else [args.accum] * args.world
    args.accum = accs[args.rank]
    ACC_TOTAL = sum(accs)
    if args.dtype_by_rank:
        args.dtype = args.dtype_by_rank.split(",")[args.rank]
else:
    ACC_TOTAL = args.accum
MAIN = args.rank == 0
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda:0")
dtype = getattr(torch, args.dtype)
torch.manual_seed(args.seed)

voc = Vocab(args.vocab_map)
vmap = torch.from_numpy(voc.vmap)
from ariautils.tokenizer import AbsTokenizer

KIND, VEL = aria_tables(AbsTokenizer())
shards = {}


def shard(i):
    if i not in shards:
        shards[i] = np.memmap(os.path.join(args.data, f"tokens_{i:03d}.bin"), dtype=np.uint16, mode="r")
    return shards[i]


recs = [json.loads(line) for line in open(os.path.join(args.data, "index.jsonl"))]
recs = [r for r in recs if "n_tok" in r and r["n_tok"] >= args.T + 1]
is_val = lambda key: zlib.crc32(key.split("_")[0].encode()) % 100 < args.val_pct
train = [r for r in recs if not is_val(r["key"])]
val = [r for r in recs if is_val(r["key"])]
tr_w = np.array([r["n_tok"] - args.T for r in train], dtype=np.float64)
tr_cdf = np.cumsum(tr_w) / tr_w.sum()
rng = np.random.default_rng(args.seed + 1000 * args.rank + int(time.time()) % 100000)
EXTRA = args.T // 2 + 600  # raw tokens past the window, so the last segments' controls see all their notes


def add_controls(raw, drop_controls):
    """Aria ids -> (compact ids with two control tokens after every <T>, mask of the control positions)."""
    tpos, dens, loud = levels(raw, KIND, VEL)
    cd, cv = voc.density(dens), voc.loudness(loud)
    if drop_controls:
        drop = rng.random(len(tpos)) < args.ctrl_dropout
        cd, cv = np.where(drop, voc.none_d, cd), np.where(drop, voc.none_v, cv)
    return voc.insert_controls(raw, cd, cv, tpos)


def window(r, off, train_window=True):
    end = min(r["n_tok"], off + args.T + 1 + EXTRA)
    a = np.asarray(shard(r["shard"])[r["offset"] + off: r["offset"] + end]).astype(np.int64)
    out, ctrl = add_controls(a, train_window and args.ctrl_dropout > 0)
    x = torch.from_numpy(out[: args.T + 1])
    y = x[1:].clone()
    y[torch.from_numpy(ctrl[1: args.T + 1])] = -100  # controls are given, not predicted
    return x, y


def batch(bs):
    ws = []
    for _ in range(bs):
        i = min(int(np.searchsorted(tr_cdf, rng.random())), len(train) - 1)
        ws.append(window(train[i], int(rng.integers(0, train[i]["n_tok"] - args.T))))
    x = torch.stack([w[0] for w in ws]).pin_memory().to(dev, non_blocking=True)
    y = torch.stack([w[1] for w in ws]).pin_memory().to(dev, non_blocking=True)
    return x[:, :-1], y


vrng = np.random.default_rng(1234)  # the same validation windows in every run
val_set = []
for i in vrng.choice(len(val), size=min(args.eval_windows, len(val)), replace=False):
    r = val[i]
    val_set.append(window(r, int(vrng.integers(0, r["n_tok"] - args.T)), train_window=False))
val_x = torch.stack([w[0] for w in val_set])
val_y = torch.stack([w[1] for w in val_set])

model = RWKV7(voc.size, args.d, args.L).to(dev)
decay, no_decay = [], []
for n, p in model.named_parameters():
    (decay if p.dim() >= 2 and "emb" not in n else no_decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=True)
scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
tok_per_step = args.bs * args.T * ACC_TOTAL
total_steps = int(args.tokens // tok_per_step)
decay_start = int(total_steps * (1 - args.decay_frac))
if DIST:
    with torch.no_grad():
        for p_ in model.parameters():
            dist.broadcast(p_.data, 0)


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    if s < decay_start:
        return args.lr
    return args.lr * (1 - math.sqrt((s - decay_start) / max(1, total_steps - decay_start)))


step = 0
ckpt_path = os.path.join(args.out, "ckpt.pt")
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if ck["scaler"] and scaler.is_enabled():
        scaler.load_state_dict(ck["scaler"])
    step = ck["step"]
    print(f"resumed at step {step}", flush=True)
elif args.init_from:
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    assert ck["vocab"] == voc.size, f"vocab {ck['vocab']} in {args.init_from} != {voc.size}"
    model.load_state_dict(ck["model"])
    if not args.reset_opt:
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["initial_lr"] = args.lr
    print(f"initialised from {args.init_from} (step {ck['step']})", flush=True)
    del ck


def save():
    if not MAIN:
        return
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "args": vars(args), "vocab": voc.size}, tmp)
    os.replace(tmp, ckpt_path)


stop = {"flag": False}
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: stop.update(flag=True))


@torch.no_grad()
def evaluate():
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_x), args.bs):
        xb, yb = val_x[i:i + args.bs].to(dev), val_y[i:i + args.bs].to(dev)
        with torch.autocast("cuda", dtype=dtype):
            tot += model(xb[:, :-1], yb).item() * len(xb)
        n += len(xb)
    model.train()
    return tot / n


if args.eval_only:
    kind_c = torch.full((voc.size,), 4, dtype=torch.long)  # by compact id
    for aid, kind in enumerate(KIND):
        if voc.vmap[aid] >= 0:
            kind_c[int(voc.vmap[aid])] = int(kind)
    names = ["note", "onset", "dur", "<T>", "other"]
    sums, cnts = torch.zeros(6), torch.zeros(6)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(val_x), args.bs):
            xb, yb = val_x[i:i + args.bs].to(dev), val_y[i:i + args.bs].to(dev)
            with torch.autocast("cuda", dtype=dtype):
                lg = model(xb[:, :-1])
            nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), yb.reshape(-1), ignore_index=-100,
                                  reduction="none")
            k = kind_c[yb.reshape(-1).clamp(min=0).cpu()]
            k[yb.reshape(-1).cpu() == -100] = 5
            sums += torch.bincount(k, weights=nll.float().cpu(), minlength=6)[:6]
            cnts += torch.bincount(k, minlength=6)[:6].float()
    res = {n: round((sums[j] / cnts[j]).item(), 4) for j, n in enumerate(names) if cnts[j] > 0}
    res["mean"] = round((sums[:5].sum() / cnts[:5].sum()).item(), 4)
    res["step"] = step
    print(json.dumps(res), flush=True)
    sys.exit(0)

log = open(os.path.join(args.out, "log.jsonl" if MAIN else f"log_rank{args.rank}.jsonl"), "a")
meta = {"event": "start", "step": step, "total_steps": total_steps, "tok_per_step": tok_per_step, "vocab": voc.size,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "train_files": len(train),
        "val_files": len(val), "args": vars(args), "time": time.time()}
log.write(json.dumps(meta) + "\n")
log.flush()
print(json.dumps(meta), flush=True)


def flat_grads():
    gs = [p_.grad if p_.grad is not None else torch.zeros_like(p_) for p_ in model.parameters()]
    return torch.cat([g.reshape(-1).float() for g in gs])


def unflat_grads(buf):
    i = 0
    for p_ in model.parameters():
        n = p_.numel()
        if p_.grad is None:
            p_.grad = torch.zeros_like(p_)
        p_.grad.copy_(buf[i:i + n].view_as(p_))
        i += n


# A rank with a non-finite loss or gradient sits the step out and the others' gradient is rescaled to a full step.
# Ten skipped updates in a row, or a bad rank 50 steps in a row, end the run as "failed".
STORM_SKIPS, STORM_BAD = 10, 50
run_skip = run_bad = 0
failed_reason = ""
stop_now = False  # with several ranks only the all-reduced stop flag may end the loop
t_ck = t0 = time.time()
tok0 = 0
n_bad, n_skip = [0] * args.world, 0
x, y = batch(args.bs)
while step < total_steps and not stop_now:
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    loss_acc = 0.0
    for micro in range(args.accum):
        with torch.autocast("cuda", dtype=dtype):
            loss = model(x, y) / ACC_TOTAL
        x, y = batch(args.bs)
        scaler.scale(loss).backward()
        loss_acc += loss.item()
    scaler.unscale_(opt)
    loss_ok = math.isfinite(loss_acc)
    if DIST:
        buf = flat_grads()
        flag = torch.zeros(1 + args.world)
        flag[0] = float(stop["flag"])
        flag[1 + args.rank] = float(not (loss_ok and bool(torch.isfinite(buf).all())))
        dist.all_reduce(flag)
        bad = [bool(flag[1 + r] > 0) for r in range(args.world)]
        if bad[args.rank]:
            buf.zero_()
        dist.all_reduce(buf)
        good = sum(a for a, b in zip(accs, bad) if not b)
        if 0 < good < ACC_TOTAL:
            buf.mul_(ACC_TOTAL / good)
        unflat_grads(buf)
        if flag[0] > 0:
            stop["flag"] = stop_now = True
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        stepped = good > 0 and math.isfinite(gn)
        if stepped:
            opt.step()
        if scaler.is_enabled():
            # only a backward overflow is the loss scale's fault; a non-finite forward leaves it unchanged
            scaler.update() if loss_ok else scaler.update(scaler.get_scale())
            if scaler.get_scale() < 1.0:
                scaler.update(1.0)
        if (step + 1) % args.resync_every == 0:
            with torch.no_grad():
                for p_ in model.parameters():
                    dist.broadcast(p_.data, 0)
    else:
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        bad = [not (loss_ok and math.isfinite(gn))]
        stepped = not bad[0]
        if scaler.is_enabled():
            scaler.step(opt)  # skips a step with non-finite gradients by itself
        elif stepped:
            opt.step()
        scaler.update()
        stop_now = stop["flag"]
    n_skip += not stepped
    n_bad = [n + b for n, b in zip(n_bad, bad)]
    if any(bad) or not stepped:
        if MAIN:
            log.write(json.dumps({"event": "nonfinite", "step": step + 1, "bad_ranks": [r for r, b in enumerate(bad) if b],
                                  "update_skipped": not stepped, "time": time.time()}) + "\n")
        run_bad = run_bad + 1 if any(bad) else 0
        run_skip = run_skip + 1 if not stepped else 0
        if run_skip >= STORM_SKIPS or run_bad >= STORM_BAD:
            failed_reason = f"{run_skip} skipped updates / {run_bad} steps with a bad rank in a row"
            stop_now = True
    else:
        run_bad = run_skip = 0
    opt.zero_grad(set_to_none=True)
    step += 1
    tok0 += tok_per_step
    if step % 10 == 0 or step == 1:
        rec = {"step": step, "loss": round(loss_acc * ACC_TOTAL / args.accum, 4), "lr": round(lr_at(step), 6),
               "gnorm": round(gn, 3), "tok_s": round(tok0 / (time.time() - t0)), "tokens": step * tok_per_step,
               "time": time.time()}
        if any(n_bad) or n_skip:
            rec.update({"bad_ranks": n_bad, "skipped": n_skip})
        n_bad, n_skip = [0] * args.world, 0
        if MAIN and (step % args.eval_every == 0 or step == total_steps):
            rec["val"] = round(evaluate(), 4)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 50 == 0 or "val" in rec:
            print(json.dumps(rec), flush=True)
        t0, tok0 = time.time(), 0
    if time.time() - t_ck > args.ckpt_minutes * 60:
        save()
        t_ck = time.time()
    if MAIN and args.milestone_every and (step % args.milestone_every == 0 or step == decay_start):
        save()
        t_ck = time.time()
        shutil.copyfile(ckpt_path, os.path.join(args.out, "ckpt_predecay.pt" if step == decay_start else f"ckpt_step{step}.pt"))
save()
final = {"event": "failed" if failed_reason else "stopped" if stop["flag"] else "done", "step": step, "time": time.time()}
if failed_reason:
    final["reason"] = failed_reason
elif not stop["flag"] and MAIN:
    final["val"] = round(evaluate(), 4)
log.write(json.dumps(final) + "\n")
print(json.dumps(final), flush=True)
