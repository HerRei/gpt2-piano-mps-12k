import importlib.util
import tempfile
import unittest
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pretty_midi",
    "torch",
    "transformers",
    "miditok",
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_SMOKE_DEPS = all(has_module(name) for name in REQUIRED_MODULES)


@unittest.skipUnless(
    HAS_SMOKE_DEPS,
    "smoke test dependencies are not installed",
)
class DependencySmokeTests(unittest.TestCase):
    def test_tokenize_and_run_tiny_forward_pass(self):
        import pretty_midi
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel

        from scripts.tokenizer_utils import build_remi_tokenizer, token_ids_array

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = Path(tmpdir) / "smoke.mid"

            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
            piano = pretty_midi.Instrument(program=0)
            notes = [60, 64, 67, 72, 71, 67, 64, 60]
            for index, pitch in enumerate(notes):
                start = index * 0.5
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=78 + (index % 3) * 6,
                        pitch=pitch,
                        start=start,
                        end=start + 0.4,
                    )
                )
            midi.instruments.append(piano)
            midi.write(str(midi_path))

            tokenizer = build_remi_tokenizer()
            ids = token_ids_array(tokenizer(str(midi_path)))
            self.assertGreater(len(ids), 8)

            config = GPT2Config(
                vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
                n_positions=128,
                n_ctx=128,
                n_embd=64,
                n_layer=2,
                n_head=2,
                use_cache=False,
            )
            model = GPT2LMHeadModel(config)

            prompt_ids = ids[: min(len(ids), 64)].tolist()
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)

            with torch.no_grad():
                output = model(input_ids=input_ids, labels=input_ids)

            self.assertTrue(torch.isfinite(output.loss).item())


if __name__ == "__main__":
    unittest.main()
