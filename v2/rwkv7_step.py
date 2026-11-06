"""RWKV-7 one-token step model in pure PyTorch, matching flash-linear-attention
(fla) 0.5.2's RWKV7ForCausalLM.

* Parameter names are fla's, so an fla state_dict loads with strict=True:
    m = RWKV7Step(RWKV7Cfg(d=768, L=12))
    m.load_fla_state_dict(fla_model.state_dict())
  (checked against the real fla 0.5.2 module tree with fla_construct.py).
* `forward` computes ONE token with every piece of state as an explicit,
  fixed-shape input and output, so Core ML can place it on the Neural Engine:
    tok        (1, 1)  int
    shift_att  (L, 1, d)        attn_norm output of the previous token (time-mix shift)
    shift_ffn  (L, 1, d)        ffn_norm output of the previous token (channel-mix shift)
    wkv        (L, H, 64, 64)   per-head matrix state, [key dim, value dim] as in fla
  -> logits (1, V), new shift_att, new shift_ffn, new wkv.
  v_first (the value residual) is recomputed from layer 0 for every token and
  needs no state. A fresh sequence starts from all-zero state (fla's default).
* `forward_sequence` is a sequence-mode reference that follows fla's own code path
  (token shift over the sequence, fused-addcmul/k-update/gate-correction reference
  formulas, the DPLR recurrence from fla/ops/generalized_delta_rule/dplr/naive.py
  with scale 1 and a=-kk, b=kk*a). Use it to check the step model.

Config (fla RWKV7Config): head_dim=64, decay_low_rank_dim=64, gate_low_rank_dim=128,
a_low_rank_dim=64, v_low_rank_dim=16, hidden_ratio=4, fuse_norm=True,
norm_first=True, norm_bias=True, norm_eps=1e-5, tie_word_embeddings=False,
hidden_act='sqrelu'.

fla details reproduced here (fla/layers/rwkv7.py, fla/models/rwkv7/modeling_rwkv7.py):
  delta = x_prev - x                      (token_shift, zero for the first token)
  x_r..x_g = x + delta * x_r..x_g          (fused_addcmul_rwkv7)
  w  = -0.6065306597126334 * sigmoid(w_lora(xw))   (log decay; decay = exp(w))
  v  = lerp(v, v_first, sigmoid(v_lora(xv)))       (layers > 0; layer 0 sets v_first = v)
  a  = sigmoid(a_lora(xa));  g = g_lora(xg)  (Linear-sigmoid-Linear, no bias)
  kk = l2_norm(k * k_k per head)  (x / sqrt(sum x^2 + 1e-6), fla's l2_norm eps)
  k  = k + k*(a-1)*k_a                     (fused_k_rwkv7)
  S  = diag(exp(w)) S + (kk*a) ((-kk)^T S) + k v^T ;  o = r^T S   (scale 1.0)
  o  = GroupNorm(H groups, eps = 64 * 1e-5, affine with bias)(o)
  o  = (o + sum(r*k*r_k per head) * v) * g  (gate_output_correction)
  o  = o_proj(o)
  block: residual = pre_norm(x) on layer 0 else x; x = residual + attn(attn_norm(residual));
         x = x + value(relu(key(h + (h_prev - h) * ffn.x_k))^2) with h = ffn_norm(x)
  logits = lm_head(norm(x))
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

W_SCALE = -0.6065306597126334  # = -exp(-0.5)


@dataclass
class RWKV7Cfg:
    d: int = 768
    L: int = 12
    vocab: int = 2816
    head_dim: int = 64
    decay_low_rank_dim: int = 64
    gate_low_rank_dim: int = 128
    a_low_rank_dim: int = 64
    v_low_rank_dim: int = 16
    hidden_ratio: int = 4
    norm_eps: float = 1e-5

    @property
    def H(self):
        return self.d // self.head_dim

    @property
    def ffn_dim(self):
        n = int(self.d * self.hidden_ratio)
        return 32 * ((n + 31) // 32)


class LoRA(nn.Module):
    """fla.layers.rwkv6.LoRA: Linear(no bias) -> act -> Linear(bias?)."""

    def __init__(self, d_in, d_out, rank, bias=True, activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), None: nn.Identity()}[activation]
        self.lora = nn.Sequential(nn.Linear(d_in, rank, bias=False), act, nn.Linear(rank, d_out, bias=bias))

    def forward(self, x):
        return self.lora(x)


class HeadGroupNorm(nn.Module):
    """fla.modules.GroupNorm(num_groups=H, hidden_size=d, bias=True): per-head layer norm + affine."""

    def __init__(self, H, d, eps):
        super().__init__()
        self.H, self.d, self.eps = H, d, eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):  # x (T, d); static shapes only, so tracing emits no shape arithmetic
        xh = x.reshape(-1, self.H, self.d // self.H)
        xh = F.layer_norm(xh, (self.d // self.H,), eps=self.eps)
        return xh.reshape(-1, self.d) * self.weight + self.bias


class Attn(nn.Module):
    def __init__(self, c: RWKV7Cfg, idx: int):
        super().__init__()
        d, H, dh = c.d, c.H, c.head_dim
        self.idx, self.H, self.dh = idx, H, dh
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, n, nn.Parameter(torch.zeros(1, 1, d)))
        self.k_k = nn.Parameter(torch.zeros(d))
        self.k_a = nn.Parameter(torch.zeros(d))
        self.r_k = nn.Parameter(torch.zeros(H, dh))
        self.r_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.w_lora = LoRA(d, d, c.decay_low_rank_dim, activation="tanh")
        if idx != 0:
            self.v_lora = LoRA(d, d, c.v_low_rank_dim, activation=None)
        self.a_lora = LoRA(d, d, c.a_low_rank_dim, activation=None)
        self.g_lora = LoRA(d, d, c.gate_low_rank_dim, activation="sigmoid", bias=False)
        self.g_norm = HeadGroupNorm(H, d, eps=dh * c.norm_eps)

    def _mix(self, h, delta, v_first):
        """Projections shared by step and sequence mode. h, delta (..., d)."""
        xr = h + delta * self.x_r[0]
        xw = h + delta * self.x_w[0]
        xk = h + delta * self.x_k[0]
        xv = h + delta * self.x_v[0]
        xa = h + delta * self.x_a[0]
        xg = h + delta * self.x_g[0]
        r = self.r_proj(xr)
        w = W_SCALE * torch.sigmoid(self.w_lora(xw))
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        if self.idx == 0:
            v_first = v
        else:
            v = v + torch.sigmoid(self.v_lora(xv)) * (v_first - v)  # torch.lerp(v, v_first, s)
        a = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)
        return r, w, k, v, a, g, v_first

    def step(self, h, x_prev, S, v_first):
        """h (1, d) = attn_norm output; x_prev (1, d); S (1, H, dh, dh); returns o (1, d), S', v_first.

        Layout chosen so the Neural Engine compiler accepts it (found by bisection on macOS 26.6,
        M1 Pro): per-head reductions (group norm, gate correction) run on rank-3 (1, H, dh)
        tensors; the same reductions on rank-4 (1, H, 1, dh) make ANECCompile() fail and Core ML
        silently runs the whole model on the CPU. The state matmuls are rank 4.
        """
        H, dh, d = self.H, self.dh, self.H * self.dh
        r, w, k, v, a, g, v_first = self._mix(h, x_prev - h, v_first)
        kk = (k * self.k_k).reshape(1, H, 1, dh)
        kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)  # fla l2_norm, eps 1e-6
        k = k + k * (a - 1) * self.k_a  # fused_k_rwkv7
        # S (1, H, K, V):  S' = diag(exp w) S + (kk*a) ((-kk)^T S) + k v^T ;  o = r^T S'
        sa = torch.matmul(-kk, S)  # (1, H, 1, V)
        S = S * torch.exp(w).reshape(1, H, dh, 1) \
            + torch.matmul((kk.reshape(1, d) * a).reshape(1, H, dh, 1), sa) \
            + torch.matmul(k.reshape(1, H, dh, 1), v.reshape(1, H, 1, dh))
        o = torch.matmul(r.reshape(1, H, 1, dh), S).reshape(1, H, dh)
        # g_norm: per-head layer norm (biased variance, eps = 64e-5) + affine
        oc = o - o.mean(-1, keepdim=True)
        o = (oc / torch.sqrt((oc * oc).mean(-1, keepdim=True) + self.g_norm.eps)).reshape(1, d)
        o = o * self.g_norm.weight + self.g_norm.bias
        # gate_output_correction: (o + sum_head(r*k*r_k) * v) * g
        corr = (r * k * self.r_k.reshape(1, d)).reshape(1, H, dh).sum(-1, keepdim=True) * v.reshape(1, H, dh)
        o = self.o_proj((o + corr.reshape(1, d)) * g)
        return o, S, v_first

    def sequence(self, h, v_first, S0=None):
        """fla-style sequence mode. h (T, d); returns o (T, d), final S (H, dh, dh), v_first (T, d)."""
        T, d = h.shape
        H, dh = self.H, self.dh
        shifted = torch.cat([h.new_zeros(1, d), h[:-1]], 0)  # nn.ZeroPad2d((0, 0, 1, -1)) / token_shift
        r, w, k, v, a, g, v_first = self._mix(h, shifted - h, v_first)
        kk = (k * self.k_k).view(T, H, dh)
        kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
        k = k.addcmul(k * (a - 1), self.k_a)  # fla k_update_ref
        r, w, k, v, a = (t.view(T, H, dh) for t in (r, w, k, v, a))
        alpha, beta = -kk, kk * a
        # dplr_recurrence (fla/ops/generalized_delta_rule/dplr/naive.py) with scale 1.0
        S = torch.zeros(H, dh, dh, dtype=h.dtype) if S0 is None else S0.clone()
        o = torch.zeros(T, H, dh, dtype=h.dtype)
        for t in range(T):
            kv = k[t][..., None] * v[t][..., None, :] + (S * alpha[t][..., None]).sum(-2, keepdim=True) * beta[t][..., None]
            S = S * w[t].exp()[..., None] + kv
            o[t] = torch.einsum("hd,hdm->hm", r[t], S)
        o = self.g_norm(o.reshape(T, d))
        corr = ((r * k * self.r_k.unsqueeze(0)).sum(-1, keepdim=True) * v).view(T, d)  # gate_output_correction_ref
        o = self.o_proj((o + corr) * g)
        return o, S, v_first


class FFN(nn.Module):
    def __init__(self, c: RWKV7Cfg):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(c.d))
        self.key = nn.Linear(c.d, c.ffn_dim, bias=False)
        self.value = nn.Linear(c.ffn_dim, c.d, bias=False)

    def forward(self, h, h_prev):
        kx = self.key(h + (h_prev - h) * self.x_k)
        r = torch.relu(kx)
        return self.value(r * r)


class Block(nn.Module):
    def __init__(self, c: RWKV7Cfg, idx: int):
        super().__init__()
        if idx == 0:
            self.pre_norm = nn.LayerNorm(c.d, eps=c.norm_eps)
        self.attn_norm = nn.LayerNorm(c.d, eps=c.norm_eps)
        self.attn = Attn(c, idx)
        self.ffn_norm = nn.LayerNorm(c.d, eps=c.norm_eps)
        self.ffn = FFN(c)


class _Model(nn.Module):
    def __init__(self, c: RWKV7Cfg):
        super().__init__()
        self.embeddings = nn.Embedding(c.vocab, c.d)
        self.layers = nn.ModuleList(Block(c, i) for i in range(c.L))
        self.norm = nn.LayerNorm(c.d, eps=c.norm_eps)


class RWKV7Step(nn.Module):
    def __init__(self, c: RWKV7Cfg):
        super().__init__()
        self.c = c
        self.model = _Model(c)
        self.lm_head = nn.Linear(c.d, c.vocab, bias=False)

    def load_fla_state_dict(self, sd):
        sd = {k: v for k, v in sd.items() if not k.endswith("x_x")}
        return self.load_state_dict(sd, strict=True)

    def zero_state(self, dtype=torch.float32):
        c = self.c
        return (torch.zeros(c.L, 1, c.d, dtype=dtype), torch.zeros(c.L, 1, c.d, dtype=dtype),
                torch.zeros(c.L, c.H, c.head_dim, c.head_dim, dtype=dtype))

    def forward(self, tok, shift_att, shift_ffn, wkv):
        return self.forward_emb(self.model.embeddings(tok).reshape(1, self.c.d), shift_att, shift_ffn, wkv)

    def forward_emb(self, x, shift_att, shift_ffn, wkv):
        """Same as forward, with the embedding row x (1, d) looked up by the caller."""
        v_first = x  # placeholder, replaced by layer 0
        na, nf, ns = [], [], []
        for i, b in enumerate(self.model.layers):
            if i == 0:
                x = b.pre_norm(x)
            h = b.attn_norm(x)
            o, S, v_first = b.attn.step(h, shift_att[i], wkv[i:i + 1], v_first)
            x = x + o
            h2 = b.ffn_norm(x)
            x = x + b.ffn(h2, shift_ffn[i])
            na.append(h)
            nf.append(h2)
            ns.append(S)
        logits = self.lm_head(self.model.norm(x))
        return logits, torch.stack(na), torch.stack(nf), torch.cat(ns, 0)

    @torch.no_grad()
    def forward_sequence(self, ids):
        """Reference: ids (T,) long -> logits (T, V), final (shift_att, shift_ffn, wkv) from zero state."""
        x = self.model.embeddings(ids)
        v_first = torch.zeros_like(x)
        na, nf, ns = [], [], []
        for b in self.model.layers:
            residual = b.pre_norm(x) if hasattr(b, "pre_norm") else x
            h = b.attn_norm(residual)
            o, S, v_first = b.attn.sequence(h, v_first)
            x = residual + o  # fused ffn_norm(o, residual, prenorm=True) returns this sum as the new residual
            h2 = b.ffn_norm(x)
            shifted = torch.cat([h2.new_zeros(1, h2.shape[1]), h2[:-1]], 0)
            x = x + b.ffn.value(torch.relu(b.ffn.key(h2.addcmul(shifted - h2, b.ffn.x_k))) ** 2)
            na.append(h[-1:])
            nf.append(h2[-1:])
            ns.append(S)
        return self.lm_head(self.model.norm(x)), (torch.stack(na), torch.stack(nf), torch.stack(ns))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def randomize_like_trained(m, seed=0, scale=0.02):
    """fla's init leaves several matrices at zero (LoRA down-projections, ffn.value); add noise
    so every path is active for numerical checks. Speed does not depend on values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return m


if __name__ == "__main__":
    # self-check: step loop == sequence reference (fp32), on a small random model
    torch.manual_seed(0)
    m = randomize_like_trained(RWKV7Step(RWKV7Cfg(d=256, L=3)), scale=0.1).eval()
    ids = torch.randint(0, 2816, (40,))
    ref, (ra, rf, rs) = m.forward_sequence(ids)
    st = m.zero_state()
    err = 0.0
    with torch.no_grad():
        for t in range(len(ids)):
            lg, *st = m(ids[t:t + 1].view(1, 1), *st)
            err = max(err, (lg[0] - ref[t]).abs().max().item())
    print("logits max abs diff over all tokens:", err, "| state diffs:",
          [round((a - b).abs().max().item(), 8) for a, b in zip(st, (ra, rf, rs))])
