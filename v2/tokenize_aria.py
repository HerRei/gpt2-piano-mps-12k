"""Tokenize Aria-MIDI with Aria's AbsTokenizer into uint16 shards and an index.

usage: python v2/tokenize_aria.py <aria-midi root> <out_dir> [--procs P]
The published model used the pruned, de-duplicated release (aria-midi-v1-pruned-ext). Writes
out_dir/tokens_XXX.bin (uint16 Aria ids), out_dir/index.jsonl (one line per file: key, shard, offset, n_tok,
n_notes, n_T, inst, or err), out_dir/token_counts.npy and out_dir/summary.json.
"""
import argparse
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
import numpy as np

tok = None


def init():
    global tok
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()


def work(path):
    from ariautils.midi import MidiDict
    key = os.path.basename(path)[:-4]
    try:
        md = MidiDict.from_midi(path)
        seq = tok.tokenize(md)
        ids = tok.encode(seq)
        n_notes = sum(1 for t in seq if isinstance(t, tuple) and t[0] in tok.instruments_wd)
        n_T = sum(1 for t in seq if t == tok.time_tok)
        insts = Counter(t[0] for t in seq if isinstance(t, tuple) and t[0] in tok.instruments_wd)
        return key, np.asarray(ids, dtype=np.uint16), {"n_notes": n_notes, "n_T": n_T, "inst": dict(insts)}, None
    except Exception as e:
        return key, None, None, f"{type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--shard_tokens", type=int, default=500_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = []
    for dp, _, fns in os.walk(a.root):
        files += [os.path.join(dp, f) for f in fns if f.endswith(".mid")]
    files.sort()
    if a.sample:
        random.Random(0).shuffle(files)
        files = files[:a.sample]
    init()
    vocab_size = len(tok.vocab)
    t0 = time.time()
    shard, offset, fh = 0, 0, open(os.path.join(a.out, "tokens_000.bin"), "wb")
    idx = open(os.path.join(a.out, "index.jsonl"), "w")
    used = np.zeros(vocab_size, dtype=np.int64)
    tot_tok = tot_notes = n_ok = n_err = 0
    with Pool(a.procs, initializer=init) as pool:
        for i, (key, ids, st, err) in enumerate(pool.imap_unordered(work, files, chunksize=64)):
            if err:
                n_err += 1
                idx.write(json.dumps({"key": key, "err": err}) + "\n")
                continue
            if offset + len(ids) > a.shard_tokens:
                fh.close()
                shard += 1
                offset = 0
                fh = open(os.path.join(a.out, f"tokens_{shard:03d}.bin"), "wb")
            fh.write(ids.tobytes())
            used += np.bincount(ids, minlength=vocab_size)
            idx.write(json.dumps({"key": key, "shard": shard, "offset": offset, "n_tok": len(ids), **st}) + "\n")
            offset += len(ids)
            tot_tok += len(ids)
            tot_notes += st["n_notes"]
            n_ok += 1
            if i % 20000 == 0:
                print(f"{i}/{len(files)} files, {tot_tok/1e9:.3f}B tokens, {time.time()-t0:.0f}s", flush=True)
    fh.close()
    idx.close()
    np.save(os.path.join(a.out, "token_counts.npy"), used)
    summary = {"files": len(files), "ok": n_ok, "err": n_err, "tokens": tot_tok, "notes": tot_notes,
               "tokens_per_note": round(tot_tok / max(tot_notes, 1), 3), "vocab_size_full": vocab_size,
               "vocab_used": int((used > 0).sum()), "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
