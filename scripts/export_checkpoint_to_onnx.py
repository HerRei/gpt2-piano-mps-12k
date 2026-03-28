import argparse
import json
import shutil
import site
from datetime import datetime, timezone
from pathlib import Path

from _paths import ROOT

VENDOR_DIR = ROOT / ".onnx_export_vendor"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "epoch_03"
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizer.json"
DEFAULT_TOKENIZER_META = ROOT / "artifacts" / "meta.json"
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "onnx" / "epoch_03"

if VENDOR_DIR.exists():
    site.addsitedir(str(VENDOR_DIR))

import numpy as np
import onnx
from onnx import checker as onnx_checker
import torch
from transformers import GPT2LMHeadModel

try:
    import onnxruntime as ort
except Exception:
    ort = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a saved checkpoint to an ONNX bundle for external inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory produced by save_pretrained().",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to the Miditok tokenizer json used for training.",
    )
    parser.add_argument(
        "--tokenizer-meta",
        type=Path,
        default=DEFAULT_TOKENIZER_META,
        help="Path to the tokenizer metadata json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the ONNX export bundle into.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dummy-seq-len",
        type=int,
        default=64,
        help="Sequence length used for the export trace.",
    )
    return parser.parse_args()


class GPT2LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]
        return logits


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def verify_with_onnxruntime(model, onnx_path: Path):
    if ort is None:
        return {
            "onnx_checker_passed": True,
            "onnxruntime_available": False,
        }

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        torch_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        ["logits"],
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        },
    )[0]

    return {
        "onnx_checker_passed": True,
        "onnxruntime_available": True,
        "torch_shape": list(torch_logits.shape),
        "onnx_shape": list(onnx_logits.shape),
        "max_abs_diff": float(np.max(np.abs(torch_logits - onnx_logits))),
        "mean_abs_diff": float(np.mean(np.abs(torch_logits - onnx_logits))),
    }


def write_readme(output_dir: Path, manifest: dict):
    verification = manifest["verification"]
    verification_line = (
        f"- ONNX Runtime verification max_abs_diff: {verification['max_abs_diff']:.8f}\n"
        if verification.get("onnxruntime_available")
        else "- ONNX Runtime verification was not run in this environment.\n"
    )
    readme = f"""Epoch 03 ONNX bundle

Files:
- model.onnx: exported causal LM graph.
- config.json: Hugging Face model config from the source checkpoint.
- generation_config.json: generation defaults from the source checkpoint.
- stats.json: epoch metrics for the source checkpoint.
- tokenizer.json: Miditok REMI tokenizer required to convert MIDI <-> token ids.
- tokenizer_meta.json: tokenizer metadata from training.
- export_manifest.json: machine-readable bundle description.

ONNX interface:
- input_ids: int64 tensor shaped [batch, sequence]
- attention_mask: int64 tensor shaped [batch, sequence]
- logits: float32 tensor shaped [batch, sequence, vocab]

Notes:
- This export does not include KV-cache inputs or outputs. Generation should
  call the model autoregressively with the full current token sequence.
- The matching tokenizer is required. The model alone is not enough.
- Source checkpoint: {manifest["source_checkpoint"]}
- Exported at: {manifest["exported_at_utc"]}
{verification_line}"""
    (output_dir / "README.txt").write_text(readme)


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    tokenizer_meta = args.tokenizer_meta.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer}")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    wrapped = GPT2LogitsOnly(model).cpu().eval()

    max_context = int(getattr(model.config, "n_positions", getattr(model.config, "n_ctx", 2048)))
    dummy_seq_len = max(1, min(args.dummy_seq_len, max_context))

    input_ids = torch.ones((1, dummy_seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.long)

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx_checker.check_model(onnx_model)
    verification = verify_with_onnxruntime(model, onnx_path)

    copy_if_exists(checkpoint / "config.json", output_dir / "config.json")
    copy_if_exists(checkpoint / "generation_config.json", output_dir / "generation_config.json")
    copy_if_exists(checkpoint / "stats.json", output_dir / "stats.json")
    shutil.copy2(tokenizer, output_dir / "tokenizer.json")
    copy_if_exists(tokenizer_meta, output_dir / "tokenizer_meta.json")

    stats_path = checkpoint / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    manifest = {
        "source_checkpoint": str(checkpoint),
        "source_epoch": stats.get("epoch"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": getattr(model.config, "model_type", None),
        "vocab_size": int(model.config.vocab_size),
        "n_positions": max_context,
        "n_layer": int(model.config.n_layer),
        "n_head": int(model.config.n_head),
        "n_embd": int(model.config.n_embd),
        "opset": args.opset,
        "dummy_seq_len": dummy_seq_len,
        "onnx_file": "model.onnx",
        "tokenizer_file": "tokenizer.json",
        "tokenizer_meta_file": "tokenizer_meta.json" if tokenizer_meta.exists() else None,
        "inputs": [
            {
                "name": "input_ids",
                "dtype": "int64",
                "shape": ["batch", "sequence"],
            },
            {
                "name": "attention_mask",
                "dtype": "int64",
                "shape": ["batch", "sequence"],
            },
        ],
        "outputs": [
            {
                "name": "logits",
                "dtype": "float32",
                "shape": ["batch", "sequence", "vocab_size"],
            }
        ],
        "generation_notes": {
            "uses_kv_cache": False,
            "decode_mode": "full-sequence autoregressive",
            "requires_matching_tokenizer": True,
        },
        "verification": verification,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(output_dir, manifest)

    print(f"exported_onnx: {onnx_path}")
    print(f"bundle_dir: {output_dir}")
    print(f"vocab_size: {model.config.vocab_size}")
    print(f"context_limit: {max_context}")
    print(f"dummy_seq_len: {dummy_seq_len}")


if __name__ == "__main__":
    main()
