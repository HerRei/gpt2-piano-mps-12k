"""Generate piano MIDI with the v2 RWKV-7 on the CPU, steered by density and loudness levels.

usage:
  python v2/generate.py --weights v2-rwkv7-piano.safetensors --plan 2,1:30 5,3:30 2,1:30 --out arc.mid
  python v2/generate.py --weights v2-rwkv7-piano.safetensors --free --seconds 60 --out free.mid
  python v2/generate.py --weights v2-rwkv7-piano.safetensors --prompt piece.mid --free --seconds 40 --out cont.mid

--plan is a list of density,loudness:seconds. Density levels (notes per second in a 5-s segment):
0 <0.5, 1 <1.5, 2 <3, 3 <5, 4 <8, 5 <12, 6 <18, 7 more. Loudness levels (mean velocity): 0 <45, 1 <55, 2 <65,
3 <75, 4 louder. --free gives "none" controls throughout. As in training, controls follow each 5-s marker, so
without a prompt the first 5 s have none and the plan's first entry covers them only in length.
--prompt continues the first --prompt_seconds of a MIDI file; the prompt keeps the controls of its real notes.
--gamma > 0 adds classifier-free guidance: a second stream sees "none" controls, logits = a + gamma * (a - b).
It costs a second model call per token.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rwkv7_step import RWKV7Cfg, RWKV7Step
from vocab import Vocab, aria_tables, levels


def parse_plan(items):
    plan = []
    for item in items:
        lv, seconds = item.split(":")
        d, v = (int(x) for x in lv.split(","))
        plan += [(d, v)] * round(float(seconds) / 5)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", nargs="+", default=[])
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--seconds", type=float, default=60, help="length of the new music with --free")
    ap.add_argument("--prompt")
    ap.add_argument("--prompt_seconds", type=float, default=20)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab_map", default=os.path.join(HERE, "vocab_map.npy"))
    a = ap.parse_args()
    if a.free == bool(a.plan):
        ap.error("give either --plan or --free")

    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer

    with safe_open(a.weights, "pt") as f:
        meta = f.metadata()
    voc = Vocab(a.vocab_map)
    assert int(meta["vocab"]) == voc.size, f"weights vocab {meta['vocab']} != {voc.size}"
    model = RWKV7Step(RWKV7Cfg(d=int(meta["d"]), L=int(meta["L"]), vocab=voc.size))
    model.load_fla_state_dict(load_file(a.weights))
    model.eval()
    torch.manual_seed(a.seed)

    tk = AbsTokenizer()
    cid = lambda t: int(voc.vmap[tk.tok_to_id[t]])
    t_id = cid("<T>")
    mask = torch.full((voc.size,), float("-inf"))
    mask[:voc.n_used] = 0
    for i, t in enumerate(tk.vocab):  # never sample special or prefix tokens
        if voc.vmap[i] >= 0 and (t in ("<E>", "<S>", "<P>", "<U>") or (isinstance(t, tuple) and t[0] == "prefix")):
            mask[voc.vmap[i]] = float("-inf")

    plan = parse_plan(a.plan) if a.plan else [None] * round(a.seconds / 5)

    def controls(i):
        c = plan[min(i, len(plan) - 1)]
        return (voc.none_d, voc.none_v) if c is None else (voc.density(c[0]), voc.loudness(c[1]))

    if a.prompt:
        raw = np.asarray(tk.encode(tk.tokenize(MidiDict.from_midi(a.prompt))), dtype=np.int64)
        if (voc.vmap[raw] < 0).any():
            raise SystemExit("the prompt uses tokens the model never saw (not a solo piano file?)")
        n_t = round(a.prompt_seconds / 5)
        tpos, dens, loud = levels(raw, *aria_tables(tk))
        if len(tpos) < n_t:
            raise SystemExit(f"the prompt is shorter than {a.prompt_seconds} s")
        out, _ = voc.insert_controls(raw, voc.density(dens), voc.loudness(loud), tpos)
        cut = tpos[n_t - 1] + 2 * n_t + 1  # through the n_t-th <T> and its two controls
        music = out[:cut].tolist()
        music[-2:] = controls(0)  # the first new segment follows the plan, not the prompt's own notes
    else:
        music = [cid(("prefix", "instrument", "piano")), cid("<S>")]
    guided = a.gamma != 0 and not a.free
    none = lambda t: voc.none_d if voc.density(0) <= t < voc.loudness(0) else voc.none_v if voc.loudness(0) <= t < voc.none_d else t

    def sample(logits):
        probs = F.softmax(logits / a.temp + mask, -1)
        sp, si = probs.sort(descending=True)
        sp = sp * (sp.cumsum(-1) - sp < a.top_p)
        return int(si[torch.multinomial(sp / sp.sum(), 1)])

    def step(tok, st):
        lg, *st = model(torch.tensor([[tok]]), *st)
        return lg[0], st

    t0 = time.time()
    with torch.no_grad():
        sa, sb = model.zero_state(), model.zero_state()
        for t in music:
            la, sa = step(t, sa)
            if guided:
                lb, sb = step(none(t), sb)
        seg, n = 0, 0
        while seg < len(plan):
            tok = sample(la + a.gamma * (la - lb) if guided else la)
            music.append(tok)
            la, sa = step(tok, sa)
            if guided:
                lb, sb = step(tok, sb)
            n += 1
            if tok == t_id:
                seg += 1
                for c in controls(seg):
                    music.append(c)
                    la, sa = step(c, sa)
                    if guided:
                        lb, sb = step(none(c), sb)

    toks = [tk.vocab[voc.aria_of[t]] for t in music if voc.aria_of[t] >= 0]
    tk.detokenize(toks + ["<E>"]).to_midi().save(a.out)
    notes = sum(1 for t in toks if isinstance(t, tuple) and t[0] == "piano")
    print(f"{notes} notes, {n} new tokens in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
