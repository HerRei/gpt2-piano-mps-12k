"""Turn a training checkpoint (weights + optimizer, ~1.1 GB) into the released weights file (fp32 safetensors).

usage: python v2/export.py runs/v2/ckpt.pt v2-rwkv7-piano.safetensors [--val 1.7493]
Tensor names are fla's (RWKV7ForCausalLM), so both model.py and rwkv7_step.py load the file.
"""
import argparse

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--val", default="", help="validation loss to record in the file's metadata")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = {k.removeprefix("m."): v.float().contiguous() for k, v in ck["model"].items()}
meta = {"d": str(ck["args"]["d"]), "L": str(ck["args"]["L"]), "vocab": str(ck["vocab"]), "step": str(ck["step"])}
if a.val:
    meta["val_loss"] = a.val
save_file(sd, a.out, metadata=meta)
print(f"step {ck['step']}, {sum(v.numel() for v in sd.values()) / 1e6:.2f}M parameters -> {a.out}")
