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
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "onnx" / "epoch_03_kv_cache"

if VENDOR_DIR.exists():
    site.addsitedir(str(VENDOR_DIR))

import numpy as np
import onnx
from onnx import checker as onnx_checker
import torch
from transformers import GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

try:
    import onnxruntime as ort
except Exception:
    ort = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to ONNX prompt/decode graphs with KV cache."
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
        help="Directory to write the ONNX KV bundle into.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--prompt-dummy-seq-len",
        type=int,
        default=64,
        help="Sequence length used for tracing the prompt graph.",
    )
    parser.add_argument(
        "--past-dummy-seq-len",
        type=int,
        default=8,
        help="Past-cache length used for tracing the decode graph.",
    )
    parser.add_argument(
        "--decode-dummy-seq-len",
        type=int,
        default=1,
        help="Current-token length used for tracing the decode graph.",
    )
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def kv_input_names(n_layer: int):
    names = []
    for layer in range(n_layer):
        names.append(f"past_key_values.{layer}.key")
        names.append(f"past_key_values.{layer}.value")
    return names


def kv_output_names(n_layer: int):
    names = []
    for layer in range(n_layer):
        names.append(f"present.{layer}.key")
        names.append(f"present.{layer}.value")
    return names


def flatten_cache(cache: DynamicCache):
    flat = []
    for layer in cache.layers:
        flat.append(layer.keys)
        flat.append(layer.values)
    return tuple(flat)


def build_dynamic_cache(config, flat_tensors):
    ddp_cache_data = []
    for layer in range(config.n_layer):
        key = flat_tensors[2 * layer]
        value = flat_tensors[2 * layer + 1]
        ddp_cache_data.append((key, value))
    return DynamicCache(ddp_cache_data=tuple(ddp_cache_data), config=config)


class PromptWithCacheWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return (outputs.logits,) + flatten_cache(outputs.past_key_values)


class DecodeWithCacheWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, *flat_past):
        cache = build_dynamic_cache(self.model.config, flat_past)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        return (outputs.logits,) + flatten_cache(outputs.past_key_values)


def export_prompt_model(model, onnx_path: Path, opset: int, prompt_seq_len: int):
    wrapper = PromptWithCacheWrapper(model).cpu().eval()
    input_ids = torch.arange(1, prompt_seq_len + 1, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones((1, prompt_seq_len), dtype=torch.long)

    output_names = ["logits", *kv_output_names(model.config.n_layer)]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "prompt_sequence"},
        "attention_mask": {0: "batch", 1: "prompt_sequence"},
        "logits": {0: "batch", 1: "prompt_sequence"},
    }
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch", 2: "prompt_sequence"}

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def export_decode_model(model, onnx_path: Path, opset: int, past_seq_len: int, decode_seq_len: int):
    wrapper = DecodeWithCacheWrapper(model).cpu().eval()
    batch = 1
    n_head = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    input_ids = torch.arange(1, decode_seq_len + 1, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones((batch, past_seq_len + decode_seq_len), dtype=torch.long)
    flat_past = []
    for _ in range(model.config.n_layer):
        flat_past.append(torch.zeros((batch, n_head, past_seq_len, head_dim), dtype=torch.float32))
        flat_past.append(torch.zeros((batch, n_head, past_seq_len, head_dim), dtype=torch.float32))

    input_names = ["input_ids", "attention_mask", *kv_input_names(model.config.n_layer)]
    output_names = ["logits", *kv_output_names(model.config.n_layer)]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "decode_sequence"},
        "attention_mask": {0: "batch", 1: "total_sequence"},
        "logits": {0: "batch", 1: "decode_sequence"},
    }
    for name in input_names[2:]:
        dynamic_axes[name] = {0: "batch", 2: "past_sequence"}
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch", 2: "total_sequence"}

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, *flat_past),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def compare_arrays(torch_array, onnx_array):
    abs_diff = np.abs(torch_array - onnx_array)
    return float(np.max(abs_diff)), float(np.mean(abs_diff))


def verify_bundle(model, prompt_onnx_path: Path, decode_onnx_path: Path):
    result = {"onnx_checker_passed": True, "onnxruntime_available": ort is not None}
    if ort is None:
        return result

    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    prompt_mask = torch.ones((1, 4), dtype=torch.long)
    next_ids = torch.tensor([[5]], dtype=torch.long)
    next_mask = torch.ones((1, 5), dtype=torch.long)

    with torch.no_grad():
        prompt_pt = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
        prompt_cache_pt = tuple(t.detach().cpu().clone() for t in flatten_cache(prompt_pt.past_key_values))
        decode_cache_in = build_dynamic_cache(
            model.config,
            tuple(t.to(model.device) for t in prompt_cache_pt),
        )
        decode_pt = model(
            input_ids=next_ids,
            attention_mask=next_mask,
            past_key_values=decode_cache_in,
            use_cache=True,
        )

    prompt_session = ort.InferenceSession(str(prompt_onnx_path), providers=["CPUExecutionProvider"])
    decode_session = ort.InferenceSession(str(decode_onnx_path), providers=["CPUExecutionProvider"])

    prompt_output_names = ["logits", *kv_output_names(model.config.n_layer)]
    prompt_outputs = prompt_session.run(
        prompt_output_names,
        {
            "input_ids": prompt_ids.numpy().astype(np.int64),
            "attention_mask": prompt_mask.numpy().astype(np.int64),
        },
    )
    prompt_logits_onnx = prompt_outputs[0]
    prompt_cache_onnx = prompt_outputs[1:]

    prompt_logits_max, prompt_logits_mean = compare_arrays(
        prompt_pt.logits.cpu().numpy(),
        prompt_logits_onnx,
    )
    prompt_cache_max = 0.0
    prompt_cache_mean_total = 0.0
    for torch_tensor, onnx_tensor in zip(prompt_cache_pt, prompt_cache_onnx, strict=True):
        max_diff, mean_diff = compare_arrays(torch_tensor.numpy(), onnx_tensor)
        prompt_cache_max = max(prompt_cache_max, max_diff)
        prompt_cache_mean_total += mean_diff
    prompt_cache_mean = prompt_cache_mean_total / max(1, len(prompt_cache_onnx))

    decode_input_names = ["input_ids", "attention_mask", *kv_input_names(model.config.n_layer)]
    decode_output_names = ["logits", *kv_output_names(model.config.n_layer)]
    decode_feed = {
        "input_ids": next_ids.numpy().astype(np.int64),
        "attention_mask": next_mask.numpy().astype(np.int64),
    }
    for name, tensor in zip(decode_input_names[2:], prompt_cache_onnx, strict=True):
        decode_feed[name] = tensor.astype(np.float32)

    decode_outputs = decode_session.run(decode_output_names, decode_feed)
    decode_logits_onnx = decode_outputs[0]
    decode_cache_onnx = decode_outputs[1:]
    decode_cache_pt = flatten_cache(decode_pt.past_key_values)

    decode_logits_max, decode_logits_mean = compare_arrays(
        decode_pt.logits.cpu().numpy(),
        decode_logits_onnx,
    )
    decode_cache_max = 0.0
    decode_cache_mean_total = 0.0
    for torch_tensor, onnx_tensor in zip(decode_cache_pt, decode_cache_onnx, strict=True):
        max_diff, mean_diff = compare_arrays(torch_tensor.cpu().numpy(), onnx_tensor)
        decode_cache_max = max(decode_cache_max, max_diff)
        decode_cache_mean_total += mean_diff
    decode_cache_mean = decode_cache_mean_total / max(1, len(decode_cache_onnx))

    result["prompt_verification"] = {
        "input_prompt_len": int(prompt_ids.shape[1]),
        "logits_shape": list(prompt_logits_onnx.shape),
        "cache_tensor_count": len(prompt_cache_onnx),
        "logits_max_abs_diff": prompt_logits_max,
        "logits_mean_abs_diff": prompt_logits_mean,
        "cache_max_abs_diff": prompt_cache_max,
        "cache_mean_abs_diff": prompt_cache_mean,
    }
    result["decode_verification"] = {
        "past_len": int(prompt_ids.shape[1]),
        "decode_len": int(next_ids.shape[1]),
        "logits_shape": list(decode_logits_onnx.shape),
        "cache_tensor_count": len(decode_cache_onnx),
        "logits_max_abs_diff": decode_logits_max,
        "logits_mean_abs_diff": decode_logits_mean,
        "cache_max_abs_diff": decode_cache_max,
        "cache_mean_abs_diff": decode_cache_mean,
    }
    return result


def write_readme(output_dir: Path, manifest: dict):
    verification = manifest["verification"]
    prompt_ver = verification.get("prompt_verification")
    decode_ver = verification.get("decode_verification")
    readme = f"""Epoch 03 ONNX KV-cache bundle

Files:
- model_prompt.onnx: prompt pass, returns logits and initial KV cache.
- model_decode.onnx: decode pass, consumes KV cache and returns updated KV cache.
- config.json: Hugging Face model config from the source checkpoint.
- generation_config.json: generation defaults from the source checkpoint.
- stats.json: epoch metrics for the source checkpoint.
- tokenizer.json: Miditok REMI tokenizer required to convert MIDI <-> token ids.
- tokenizer_meta.json: tokenizer metadata from training.
- export_manifest.json: machine-readable bundle description.

Workflow:
1. Run model_prompt.onnx on the full prompt.
2. Take its present.* outputs as the initial cache.
3. Run model_decode.onnx for each next token, feeding back present.* as past_key_values.*.

Prompt model inputs:
- input_ids: int64 [batch, prompt_sequence]
- attention_mask: int64 [batch, prompt_sequence]

Decode model inputs:
- input_ids: int64 [batch, decode_sequence]
- attention_mask: int64 [batch, total_sequence]
- past_key_values.<layer>.key/value: float32 [batch, heads, past_sequence, head_dim]

Outputs:
- logits: float32 [batch, sequence, vocab]
- present.<layer>.key/value: float32 [batch, heads, total_sequence, head_dim]

Source checkpoint: {manifest["source_checkpoint"]}
Exported at: {manifest["exported_at_utc"]}
Prompt verification max_abs_diff: {prompt_ver['logits_max_abs_diff']:.8f} logits / {prompt_ver['cache_max_abs_diff']:.8f} cache
Decode verification max_abs_diff: {decode_ver['logits_max_abs_diff']:.8f} logits / {decode_ver['cache_max_abs_diff']:.8f} cache
"""
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
    prompt_onnx_path = output_dir / "model_prompt.onnx"
    decode_onnx_path = output_dir / "model_decode.onnx"

    model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    max_context = int(getattr(model.config, "n_positions", getattr(model.config, "n_ctx", 2048)))
    prompt_seq_len = max(1, min(args.prompt_dummy_seq_len, max_context))
    past_seq_len = max(1, min(args.past_dummy_seq_len, max_context - 1))
    decode_seq_len = max(1, min(args.decode_dummy_seq_len, max_context - past_seq_len))

    export_prompt_model(model, prompt_onnx_path, args.opset, prompt_seq_len)
    export_decode_model(model, decode_onnx_path, args.opset, past_seq_len, decode_seq_len)

    onnx_checker.check_model(onnx.load(str(prompt_onnx_path)))
    onnx_checker.check_model(onnx.load(str(decode_onnx_path)))
    verification = verify_bundle(model, prompt_onnx_path, decode_onnx_path)

    copy_if_exists(checkpoint / "config.json", output_dir / "config.json")
    copy_if_exists(checkpoint / "generation_config.json", output_dir / "generation_config.json")
    copy_if_exists(checkpoint / "stats.json", output_dir / "stats.json")
    shutil.copy2(tokenizer, output_dir / "tokenizer.json")
    copy_if_exists(tokenizer_meta, output_dir / "tokenizer_meta.json")

    stats_path = checkpoint / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    head_dim = model.config.n_embd // model.config.n_head

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
        "head_dim": int(head_dim),
        "opset": args.opset,
        "prompt_model": {
            "file": "model_prompt.onnx",
            "inputs": [
                {"name": "input_ids", "dtype": "int64", "shape": ["batch", "prompt_sequence"]},
                {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "prompt_sequence"]},
            ],
            "outputs": [
                {"name": "logits", "dtype": "float32", "shape": ["batch", "prompt_sequence", "vocab_size"]},
                {"name": "present.<layer>.key/value", "dtype": "float32", "shape": ["batch", "heads", "prompt_sequence", "head_dim"]},
            ],
        },
        "decode_model": {
            "file": "model_decode.onnx",
            "inputs": [
                {"name": "input_ids", "dtype": "int64", "shape": ["batch", "decode_sequence"]},
                {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "total_sequence"]},
                {"name": "past_key_values.<layer>.key/value", "dtype": "float32", "shape": ["batch", "heads", "past_sequence", "head_dim"]},
            ],
            "outputs": [
                {"name": "logits", "dtype": "float32", "shape": ["batch", "decode_sequence", "vocab_size"]},
                {"name": "present.<layer>.key/value", "dtype": "float32", "shape": ["batch", "heads", "total_sequence", "head_dim"]},
            ],
        },
        "workflow": [
            "Run model_prompt.onnx on the initial prompt to build the first cache.",
            "Map each present.<layer>.key/value tensor to the matching past_key_values.<layer>.key/value input.",
            "For each decode step, pass the new token ids, a total-length attention mask, and the previous cache tensors into model_decode.onnx.",
        ],
        "verification": verification,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(output_dir, manifest)

    print(f"prompt_model: {prompt_onnx_path}")
    print(f"decode_model: {decode_onnx_path}")
    print(f"bundle_dir: {output_dir}")
    print(f"vocab_size: {model.config.vocab_size}")
    print(f"context_limit: {max_context}")


if __name__ == "__main__":
    main()
