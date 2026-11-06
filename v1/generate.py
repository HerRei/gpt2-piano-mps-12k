"""Continue a piano MIDI file with the v1 GPT-2.

usage: python v1/generate.py --weights v1-gpt2-piano.safetensors --prompt piece.mid --out continued.mid
       python v1/generate.py --weights v1-gpt2-piano.safetensors --out from-scratch.mid
       python v1/generate.py --weights v1-gpt2-piano.safetensors --prompt piece.mid --prompt_seconds 20 --seconds 40 \
           --out continued.mid

The prompt is tokenized with the same REMI settings as the training data and cut to its last --prompt_tokens
tokens; the model adds --new_tokens more (prompt and continuation together stay within the 2,048-token window).
With --prompt_seconds the prompt is instead the notes that start in its first N seconds, and with --seconds the
model keeps going until the new music is that long, sliding the 2,048-token window along (the last 1,536 tokens
stay in view, 512 new ones per round). The output file holds the prompt followed by the continuation.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from symusic import Score
from transformers import GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import ids_of, remi
from train import gpt2_config

BLOCK, KEEP = 2048, 1536


def last_onset(tok, ids):
    notes = tok.decode([ids]).to("second").tracks[0].notes
    return max((n.time for n in notes), default=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", help="MIDI file to continue; without it the model starts from a bar line")
    ap.add_argument("--prompt_tokens", type=int, default=512)
    ap.add_argument("--prompt_seconds", type=float, help="use the notes that start in the prompt's first N seconds")
    ap.add_argument("--new_tokens", type=int, default=1024)
    ap.add_argument("--seconds", type=float, help="generate until this many seconds of new music")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.seconds is None and (a.prompt_tokens if a.prompt else 1) + a.new_tokens > BLOCK:
        ap.error("prompt and --new_tokens must stay within 2048 tokens")
    if a.prompt_seconds is not None and not a.prompt:
        ap.error("--prompt_seconds needs --prompt")

    tok = remi()
    model = GPT2LMHeadModel(gpt2_config(len(tok), BLOCK))
    missing, unexpected = model.load_state_dict(load_file(a.weights), strict=False)
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise SystemExit(f"weights do not match the v1 model: missing {missing}, unexpected {unexpected}")
    model.tie_weights()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    torch.manual_seed(a.seed)

    if a.prompt_seconds is not None:
        score = Score(a.prompt).to("second").clip(0, a.prompt_seconds, clip_end=False).to("tick")
        ids = ids_of(tok, score).tolist()
        if len(ids) > KEEP:
            raise SystemExit(f"the first {a.prompt_seconds} s are {len(ids)} tokens; at most {KEEP} fit")
        start = a.prompt_seconds
    elif a.prompt:
        ids = ids_of(tok, a.prompt).tolist()[-a.prompt_tokens:]
        start = last_onset(tok, ids)
    else:
        ids = [tok.vocab["Bar_None"]]
        start = 0.0
    n_prompt = len(ids)

    def extend(ctx, n):
        x = torch.tensor([ctx], device=dev)
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=a.temp,
                                 top_p=a.top_p, max_new_tokens=n, pad_token_id=tok.pad_token_id)
        return out[0, len(ctx):].tolist()

    if a.seconds is None:
        ids += extend(ids, a.new_tokens)
        score = tok.decode([ids])
    else:
        end = start + a.seconds
        while last_onset(tok, ids) < end:
            ids += extend(ids[-KEEP:], BLOCK - KEEP)
        score = tok.decode([ids]).to("second").clip(0, end, clip_end=False).to("tick")
    score.dump_midi(a.out)
    print(f"{n_prompt} prompt tokens + {len(ids) - n_prompt} new -> {a.out}")


if __name__ == "__main__":
    main()
