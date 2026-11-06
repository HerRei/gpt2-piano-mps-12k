"""Continue a piano MIDI file with the v1 GPT-2.

usage: python v1/generate.py --weights v1-gpt2-piano.safetensors --prompt piece.mid --out continued.mid
       python v1/generate.py --weights v1-gpt2-piano.safetensors --out from-scratch.mid

The prompt is tokenized with the same REMI settings as the training data and cut to its last --prompt_tokens
tokens; the model adds --new_tokens more (prompt and continuation together stay within the 2,048-token window).
The output file holds the prompt followed by the continuation.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import ids_of, remi
from train import gpt2_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", help="MIDI file to continue; without it the model starts from a bar line")
    ap.add_argument("--prompt_tokens", type=int, default=512)
    ap.add_argument("--new_tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if (a.prompt_tokens if a.prompt else 1) + a.new_tokens > 2048:
        ap.error("prompt and --new_tokens must stay within 2048 tokens")

    tok = remi()
    model = GPT2LMHeadModel(gpt2_config(len(tok), 2048))
    missing, unexpected = model.load_state_dict(load_file(a.weights), strict=False)
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise SystemExit(f"weights do not match the v1 model: missing {missing}, unexpected {unexpected}")
    model.tie_weights()
    model.eval()
    torch.manual_seed(a.seed)

    ids = ids_of(tok, a.prompt).tolist()[-a.prompt_tokens:] if a.prompt else [tok.vocab["Bar_None"]]
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                             do_sample=True, temperature=a.temp, top_p=a.top_p, max_new_tokens=a.new_tokens,
                             pad_token_id=tok.pad_token_id)
    tok.decode([out[0].tolist()]).dump_midi(a.out)
    print(f"{len(ids)} prompt tokens + {out.shape[1] - len(ids)} new -> {a.out}")


if __name__ == "__main__":
    main()
