"""RWKV-7 for training: flash-linear-attention (fla) 0.5.2's RWKV7ForCausalLM with two changes for AMD GPUs.

1. fla calls its chunked WKV kernel with chunk size 64, tuned for NVIDIA tensor cores. On an RX 9060 XT and a
   Radeon VII chunk size 16 is 4-6x faster (B8 T1024 H12 D64, forward + backward).
2. Under fp16 autocast the WKV output can exceed fp16's range (up to 335,000 on ~1 % of windows after ~6k steps,
   against a maximum of 65,504), so the kernel runs in fp32 and the group norm after it casts back to fp16.
   bf16 has the range and is left alone.
"""
import torch
import torch.nn as nn


def _patch_fla():
    import fla.layers.rwkv7 as l7
    if getattr(l7, "_piano_patched", False):
        return
    orig = l7.chunk_rwkv7

    def up(t):
        return t.float() if torch.is_tensor(t) and t.dtype == torch.float16 else t

    def chunk(*args, **kw):
        kw["chunk_size"] = 16
        if any(torch.is_tensor(t) and t.dtype == torch.float16 for t in (*args, *kw.values())):
            args, kw = tuple(up(t) for t in args), {k: up(v) for k, v in kw.items()}
        return orig(*args, **kw)

    l7.chunk_rwkv7 = chunk
    l7._piano_patched = True


def _gnorm_to_fp16(mod, inp, out):
    if out.dtype == torch.float32 and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16:
        return out.half()


class RWKV7(nn.Module):
    def __init__(self, vocab, d=768, L=12, head_dim=64):
        super().__init__()
        _patch_fla()
        from fla.models import RWKV7Config, RWKV7ForCausalLM
        cfg = RWKV7Config(hidden_size=d, num_hidden_layers=L, head_dim=head_dim, vocab_size=vocab, fuse_norm=True,
                          fuse_cross_entropy=True, use_l2warp=False, tie_word_embeddings=False, bos_token_id=0,
                          eos_token_id=1)
        self.m = RWKV7ForCausalLM(cfg)
        for n, mod in self.m.named_modules():
            if n.endswith("attn.g_norm"):
                mod.register_forward_hook(_gnorm_to_fp16)

    def forward(self, idx, targets=None):
        if targets is None:
            return self.m(input_ids=idx).logits
        # targets are already shifted; fla shifts labels by one itself, so prepend a column it will drop
        labels = torch.cat([idx[:, :1], targets[:, :-1]], dim=1)
        return self.m(input_ids=idx, labels=labels).loss
