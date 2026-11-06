"""Convert the v2 step model to Core ML and time it on the Apple Neural Engine (macOS 15 or later).

usage: python v2/coreml.py --weights v2-rwkv7-piano.safetensors [--precision int8|fp16] [--units ne|cpu|gpu]
Needs coremltools. The embedding lookup stays on the host; the rest of one token step (all 12 layers and the
output head) is a fixed-shape Core ML program, which is what lets it run on the Neural Engine. The script
checks the converted model against the fp32 PyTorch step model, then times one stream and two streams
(classifier-free guidance needs two calls per token).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rwkv7_step import RWKV7Cfg, RWKV7Step
from vocab import Vocab


class HostEmb(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, shift_att, shift_ffn, wkv):
        return self.m.forward_emb(x, shift_att, shift_ffn, wkv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precision", choices=["int8", "fp16"], default="int8")
    ap.add_argument("--units", choices=["ne", "cpu", "gpu"], default="ne")
    ap.add_argument("--out", default="", help="mlpackage path (default: next to the weights)")
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()

    import coremltools as ct
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

    with safe_open(a.weights, "pt") as f:
        meta = f.metadata()
    c = RWKV7Cfg(d=int(meta["d"]), L=int(meta["L"]), vocab=int(meta["vocab"]))
    m = RWKV7Step(c)
    m.load_fla_state_dict(load_file(a.weights))
    m.eval()
    emb = m.model.embeddings.weight.detach().numpy().astype(np.float16)

    path = a.out or os.path.splitext(a.weights)[0] + f"-{a.precision}.mlpackage"
    if not os.path.exists(path):
        ex = (torch.randn(1, c.d) * 0.02, torch.zeros(c.L, 1, c.d), torch.zeros(c.L, 1, c.d),
              torch.zeros(c.L, c.H, c.head_dim, c.head_dim))
        names = ["x", "shift_att", "shift_ffn", "wkv"]
        traced = torch.jit.trace(HostEmb(m).eval(), ex)
        f16 = np.float16
        ml = ct.convert(traced, inputs=[ct.TensorType(n, tuple(t.shape), dtype=f16) for n, t in zip(names, ex)],
                        outputs=[ct.TensorType(o, dtype=f16) for o in ("logits", "o_shift_att", "o_shift_ffn", "o_wkv")],
                        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16)
        if a.precision == "int8":
            ml = linear_quantize_weights(ml, OptimizationConfig(
                global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
        ml.save(path)
    units = {"ne": ct.ComputeUnit.CPU_AND_NE, "cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU}
    ml = ct.models.MLModel(path, compute_units=units[a.units])

    def zero():
        return {"shift_att": np.zeros((c.L, 1, c.d), np.float16), "shift_ffn": np.zeros((c.L, 1, c.d), np.float16),
                "wkv": np.zeros((c.L, c.H, c.head_dim, c.head_dim), np.float16)}

    def run(feed, tok):
        out = ml.predict(dict(feed, x=emb[tok][None]))
        return out["logits"][0].astype(np.float32), {"shift_att": out["o_shift_att"], "shift_ffn": out["o_shift_ffn"],
                                                     "wkv": out["o_wkv"]}

    # check against the fp32 step model on a sequence it generates greedily
    from ariautils.tokenizer import AbsTokenizer

    start = int(Vocab(os.path.join(HERE, "vocab_map.npy")).vmap[AbsTokenizer().tok_to_id["<S>"]])
    toks, agree, st, feed = [start], 0, m.zero_state(), zero()
    with torch.no_grad():
        for _ in range(128):
            ref, *st = m(torch.tensor([[toks[-1]]]), *st)
            lg, feed = run(feed, toks[-1])
            agree += int(lg.argmax() == int(ref[0].argmax()))
            toks.append(int(ref[0].argmax()))
    print(f"top-1 agreement with fp32 PyTorch: {agree}/128")

    fa, fb, one, two = zero(), zero(), [], []
    for i in range(50 + a.steps):
        t0 = time.perf_counter()
        _, fa = run(fa, toks[i % len(toks)])
        t1 = time.perf_counter()
        _, fb = run(fb, toks[(i + 7) % len(toks)])
        t2 = time.perf_counter()
        if i >= 50:
            one.append(t1 - t0)
            two.append(t2 - t0)
    for name, ts in (("one stream", one), ("two streams", two)):
        ms = np.sort(np.asarray(ts) * 1000)
        print(f"{name}: median {np.median(ms):.2f} ms, p99 {np.percentile(ms, 99):.2f} ms per token")


if __name__ == "__main__":
    main()
