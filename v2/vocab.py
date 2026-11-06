"""v2 token ids: Aria's AbsTokenizer vocabulary, trimmed to the ids that occur in the training data, plus
15 control tokens (8 density levels, 5 loudness levels, and a "none" token for each).

After every 5-s marker <T> come two control tokens describing the next 5 s: its density level (notes per second)
and its loudness level (mean velocity). In training they are computed from the real notes; in generation the
caller chooses them. They are inputs only, never predicted.
"""
import numpy as np

DENS_EDGES = np.array([0.5, 1.5, 3, 5, 8, 12, 18])  # notes per second -> density level 0..7
VEL_EDGES = np.array([45, 55, 65, 75])  # mean velocity -> loudness level 0..4
N_DENS, N_VEL = len(DENS_EDGES) + 1, len(VEL_EDGES) + 1


class Vocab:
    def __init__(self, vmap_path):
        self.vmap = np.load(vmap_path)  # Aria id -> compact id, -1 if unused
        self.n_used = int((self.vmap >= 0).sum())
        self.ctrl_base = self.n_used
        self.none_d = self.ctrl_base + N_DENS + N_VEL
        self.none_v = self.none_d + 1
        self.size = ((self.none_v + 1 + 127) // 128) * 128  # padded to a multiple of 128
        self.aria_of = np.full(self.size, -1, dtype=np.int64)
        self.aria_of[self.vmap[self.vmap >= 0]] = np.nonzero(self.vmap >= 0)[0]

    def density(self, level):
        return self.ctrl_base + level

    def loudness(self, level):
        return self.ctrl_base + N_DENS + level

    def insert_controls(self, raw, dens, loud, tpos):
        """Aria ids -> compact ids with a density and a loudness token after each <T> (positions tpos)."""
        pos = np.repeat(tpos + 1, 2)
        vals = np.stack([dens, loud], 1).reshape(-1)
        out = np.insert(self.vmap[raw], pos, vals)
        ctrl = np.insert(np.zeros(len(raw), dtype=bool), pos, True)
        return out, ctrl


def aria_tables(tokenizer):
    """Per Aria id: kind (0 note, 1 onset, 2 duration, 3 <T>, 4 other) and the velocity of note tokens."""
    kind = np.full(len(tokenizer.vocab), 4, dtype=np.int8)
    vel = np.zeros(len(tokenizer.vocab), dtype=np.float32)
    for i, t in enumerate(tokenizer.vocab):
        if t == "<T>":
            kind[i] = 3
        elif isinstance(t, tuple) and t[0] == "piano":
            kind[i], vel[i] = 0, t[2]
        elif isinstance(t, tuple) and t[0] in ("onset", "dur"):
            kind[i] = 1 if t[0] == "onset" else 2
    return kind, vel


def levels(raw, kind, vel):
    """Density and loudness level of the 5 s after each <T> in a sequence of Aria ids, from its notes."""
    k = kind[raw]
    seg = np.cumsum(k == 3)  # a <T> starts the next segment
    is_note = k == 0
    nseg = seg[-1] + 1
    cnt = np.bincount(seg[is_note], minlength=nseg)
    vsum = np.bincount(seg[is_note], weights=vel[raw[is_note]], minlength=nseg)
    tpos = np.nonzero(k == 3)[0]
    sid = seg[tpos]
    dens = np.digitize(cnt[sid] / 5.0, DENS_EDGES)
    loud = np.digitize(np.where(cnt[sid] > 0, vsum[sid] / np.maximum(cnt[sid], 1), 0), VEL_EDGES)
    return tpos, dens, loud
