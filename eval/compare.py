"""Compare generated piano music with real playing, using statistics of the notes.

usage: python eval/compare.py --real real/ --system pass2=gen/ [--system pass1=gen1/ ...] [--bootstrap 500]

real/ holds the real recordings as <id>.mid; each system directory holds continuations of the same pieces as
<id>_<anything>.mid (several samples per piece are fine). Only seconds --start to --end are compared, which for
continuations of a 20-s prompt is the generated part. A logistic classifier on 22 features (key fit, notes outside
the key, interval and rhythm variety, repeated figures, loudness spread, ...) tries to tell the real recording
from the system, scored by cross-validation grouped by piece. Reported per system:
  AUC   0.5 means the classifier cannot tell them apart
  win   share of pieces where the system's version scores as more "real" than the real one
  shift the mean absolute difference of the features from real playing, in standard deviations
Damaged copies of the real pieces (shuffled or random pitches, random timing, 5 % and 15 % wrong notes) are
scored the same way, so the scale has known points. With --bootstrap N it adds 95 % intervals over pieces.
"""
import argparse
import glob
import json
import os

import mido
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KS = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KM = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJ = np.array([0, 2, 4, 5, 7, 9, 11])
T0, T1 = 20.0, 60.0  # set from --start/--end


def read(path):
    m = mido.MidiFile(path); t = 0.0; on = {}; out = []
    for msg in m:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            s, v = on.pop(msg.note); out.append((s, t - s, msg.note, v))
    a = np.array(sorted(out), dtype=float).reshape(-1, 4)
    return a


def window(a):
    return a[(a[:, 0] >= T0) & (a[:, 0] < T1)].copy()


def best_key(a):
    h = np.zeros(12)
    for s, d, p, v in a: h[int(p) % 12] += max(d, 0.05)
    best, bk = -2, None
    for k, base in (("M", KS), ("m", KM)):
        for r in range(12):
            c = np.corrcoef(h, np.roll(base, r))[0, 1]
            if c > best: best, bk = c, (k, r)
    return best, bk


def ent(x):
    x = np.asarray(x, float); x = x[x > 0]; p = x / x.sum()
    return float(-(p * np.log2(p)).sum())


def feats(a):
    if len(a) < 30: return None
    on, du, pi, ve = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    f = {}
    h = np.bincount(pi.astype(int) % 12, minlength=12); f["pc_entropy"] = ent(h)
    kf, (mode, tonic) = best_key(a); f["key_fit"] = kf
    scale = {(tonic + x) % 12 for x in (MAJ if mode == "M" else np.array([0, 2, 3, 5, 7, 8, 10]))}
    f["off_key"] = float(np.mean([int(p) % 12 not in scale for p in pi]))
    iv = np.diff(pi); f["abs_interval"] = float(np.abs(iv).mean()); f["leap_frac"] = float((np.abs(iv) > 7).mean())
    f["huge_leap"] = float((np.abs(iv) > 14).mean())
    f["interval_entropy"] = ent(np.bincount(np.clip(iv + 24, 0, 48).astype(int)))
    ioi = np.diff(on); ioi = ioi[ioi > 0.015]
    f["ioi_logsd"] = float(np.log(ioi).std()); f["ioi_entropy"] = ent(np.histogram(np.log(ioi), bins=24)[0])
    # pulse: how many onsets sit near a common grid (best of a few grids)
    best = 0
    for g in np.arange(0.10, 0.50, 0.01):
        r = (on / g) % 1; d = np.minimum(r, 1 - r) * g
        best = max(best, float((d < 0.03).mean()))
    f["grid_fit"] = best
    f["pitch_range"] = float(np.percentile(pi, 98) - np.percentile(pi, 2)); f["pitch_sd"] = float(pi.std())
    f["vel_mean"] = float(ve.mean()); f["vel_sd"] = float(ve.std())
    f["vel_lag1"] = float(np.corrcoef(ve[:-1], ve[1:])[0, 1]) if len(ve) > 3 else 0.0
    f["dur_logmean"] = float(np.log(np.clip(du, 0.02, 20)).mean())
    # polyphony: mean notes sounding at each onset
    ends = on + du; f["polyphony"] = float(np.mean([((on <= t) & (ends > t)).sum() for t in on[:: max(1, len(on) // 200)]]))
    seg = np.floor((on - T0) / 5).astype(int); cnt = np.bincount(seg, minlength=8)[:8]
    f["nps"] = len(on) / (T1 - T0); f["nps_cv"] = float(cnt.std() / max(cnt.mean(), 1e-6))
    # motif recurrence: share of 5-grams of (interval, coarse ioi) that occurred earlier in the piece
    toks = list(zip(iv.astype(int)[1:], np.clip(np.round(np.log2(np.maximum(np.diff(on)[1:], 0.02)) * 1.5), -8, 8).astype(int)))
    seen, rep = set(), 0
    for i in range(len(toks) - 4):
        g = tuple(toks[i:i + 5]); rep += g in seen; seen.add(g)
    f["motif_rec"] = rep / max(1, len(toks) - 4)
    # key stability: key fit on the first vs second half and whether the key stays the same
    h1 = a[a[:, 0] < (T0 + T1) / 2]; h2 = a[a[:, 0] >= (T0 + T1) / 2]
    if len(h1) > 20 and len(h2) > 20:
        k1, kk1 = best_key(h1); k2, kk2 = best_key(h2); f["key_same"] = float(kk1 == kk2); f["key_fit_min"] = min(k1, k2)
    else: f["key_same"] = 0.0; f["key_fit_min"] = 0.0
    return f


def negatives(a, rng):
    """controls built from a real window: each destroys one musical property."""
    out = {}
    b = a.copy(); b[:, 2] = rng.permutation(b[:, 2]); out["neg_pitch_shuffle"] = b
    b = a.copy(); b[:, 0] = np.sort(rng.uniform(T0, T1, len(b))); out["neg_time_random"] = b
    b = a.copy(); b[:, 2] = rng.integers(int(np.percentile(a[:, 2], 2)), int(np.percentile(a[:, 2], 98)) + 1, len(b)); out["neg_random_pitch"] = b
    b = a.copy(); k = rng.random(len(b)) < 0.15; b[k, 2] += rng.choice([-2, -1, 1, 2], k.sum()); out["neg_wrong_notes_15pct"] = b
    b = a.copy(); k = rng.random(len(b)) < 0.05; b[k, 2] += rng.choice([-2, -1, 1, 2], k.sum()); out["neg_wrong_notes_5pct"] = b
    return out


def cv_scores(Xh, Xs, groups):
    """grouped CV; returns AUC and paired win rate (system version scored as more human than the real one)."""
    X = np.vstack([Xh, Xs]); y = np.r_[np.ones(len(Xh)), np.zeros(len(Xs))]; g = np.r_[groups[:len(Xh)], groups[len(Xh):]]
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X[tr], y[tr])
        sc[te] = m.decision_function(X[te])
    auc = roc_auc_score(y, sc)
    sh, ss = sc[:len(Xh)], sc[len(Xh):]
    gh = np.asarray(groups[:len(Xh)]); gs = np.asarray(groups[len(Xh):])
    w = [np.mean(ss[gs == p] > sh[gh == p][0]) for p in np.unique(gs) if (gh == p).any()]
    return auc, float(np.mean(w)), sc


def load_system(d):
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "*.mid"))):
        pid = os.path.basename(p).split("_")[0]
        w = window(read(p))
        f = feats(w) if len(w) else None
        if f:
            res.setdefault(pid, []).append(f)
    return res


def main():
    global T0, T1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--system", action="append", default=[], help="name=directory, repeatable")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    T0, T1 = a.start, a.end
    rng = np.random.default_rng(0)

    real, systems = {}, {}
    for p in sorted(glob.glob(os.path.join(a.real, "*.mid"))):
        pid = os.path.splitext(os.path.basename(p))[0]
        w = window(read(p))
        f = feats(w)
        if not f:
            continue
        real[pid] = f
        for k, b in negatives(w, rng).items():
            nf = feats(b)
            if nf:
                systems.setdefault(k, {}).setdefault(pid, []).append(nf)
    for spec in a.system:
        name, d = spec.split("=", 1)
        systems[name] = load_system(d)

    names = sorted(next(iter(real.values())))
    vec = lambda f: [f[n] if np.isfinite(f[n]) else 0.0 for n in names]
    pids = sorted(real)
    ref = np.array([vec(real[p]) for p in pids])
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    rows = []
    for sysn, d in systems.items():
        common = [p for p in pids if p in d]
        if len(common) < 10:
            continue
        Xh = np.array([vec(real[p]) for p in common])
        Xs, gs = [], []
        for p in common:
            for f in d[p]:
                Xs.append(vec(f))
                gs.append(p)
        Xs, gs = np.array(Xs), np.array(gs)
        auc, win, sc = cv_scores(Xh, Xs, list(common) + list(gs))
        shift = (Xs.mean(0) - mu) / sd
        row = {"system": sysn, "pieces": len(common), "auc": round(auc, 3), "win": round(win, 3),
               "mean_abs_shift": round(float(np.abs(shift).mean()), 3),
               "largest_shifts": {n: round(float(s), 2) for n, s in sorted(zip(names, shift), key=lambda x: -abs(x[1]))[:4]}}
        if a.bootstrap:
            sh, ss, boot = sc[:len(common)], sc[len(common):], []
            for _ in range(a.bootstrap):
                idx = rng.integers(0, len(common), len(common))
                h = sh[idx]
                s = np.concatenate([ss[gs == common[i]] for i in idx])
                boot.append(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(s))], np.r_[h, s]))
            row["auc_95"] = [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])]
        rows.append(row)
    rows.sort(key=lambda r: r["auc"])
    print(f"{'system':26}{'pieces':>7}{'AUC':>7}{'win %':>7}{'shift':>7}  95 % interval")
    for r in rows:
        ci = "{:.2f}-{:.2f}".format(*r["auc_95"]) if "auc_95" in r else ""
        print(f"{r['system']:26}{r['pieces']:7d}{r['auc']:7.2f}{100 * r['win']:7.0f}{r['mean_abs_shift']:7.2f}  {ci}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
