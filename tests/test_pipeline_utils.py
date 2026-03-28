import unittest
from pathlib import Path

from scripts.pipeline_utils import (
    build_generation_command,
    checkpoint_dir,
    load_prompt_profiles,
    normalize_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


class PipelineUtilsTests(unittest.TestCase):
    def test_normalize_epoch(self):
        self.assertEqual(normalize_epoch(2), 2)
        self.assertEqual(normalize_epoch("04"), 4)
        self.assertEqual(normalize_epoch("epoch_02"), 2)

    def test_checkpoint_dir(self):
        self.assertEqual(checkpoint_dir(ROOT, 4), ROOT / "checkpoints" / "epoch_04")

    def test_load_prompt_profiles(self):
        profiles = load_prompt_profiles(ROOT / "configs" / "generation_prompt_profiles.json")
        self.assertIn("schubert_lyrical", profiles)
        self.assertEqual(profiles["schubert_lyrical"].prompt_index, 1)

    def test_build_generation_command_with_profile_prompt(self):
        profiles = load_prompt_profiles(ROOT / "configs" / "generation_prompt_profiles.json")
        profile = profiles["recital_compact"]
        command = build_generation_command(
            python_bin="python3",
            root=ROOT,
            checkpoint=ROOT / "checkpoints" / "epoch_02",
            output_dir=ROOT / "exports" / "pipeline_runs" / "test",
            run_name="epoch02_recital_compact",
            profile=profile,
            seed=123,
        )

        self.assertIn("--prompt-index", command)
        self.assertIn("2", command)
        self.assertIn("--prompt-position", command)
        self.assertIn("middle", command)

    def test_build_generation_command_with_external_midi(self):
        profiles = load_prompt_profiles(ROOT / "configs" / "generation_prompt_profiles.json")
        profile = profiles["schubert_lyrical"]
        command = build_generation_command(
            python_bin="python3",
            root=ROOT,
            checkpoint=ROOT / "checkpoints" / "epoch_04",
            output_dir=ROOT / "exports" / "pipeline_runs" / "test",
            run_name="epoch04_custom",
            profile=profile,
            seed=321,
            prompt_midi=ROOT / "seed.mid",
        )

        self.assertIn("--prompt-midi", command)
        self.assertNotIn("--prompt-index", command)


if __name__ == "__main__":
    unittest.main()
