import unittest

from scripts.generation_utils import (
    longest_run,
    parse_int_list,
    parse_str_list,
    repeated_ngram_ratio,
    slice_prompt,
)


class GenerationUtilsTests(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list(None), [])
        self.assertEqual(parse_int_list("1, 2,3"), [1, 2, 3])

    def test_parse_str_list(self):
        self.assertEqual(parse_str_list(None), [])
        self.assertEqual(parse_str_list("a, b ,c"), ["a", "b", "c"])

    def test_slice_prompt_start(self):
        prompt, meta = slice_prompt(
            full_prompt_ids=list(range(10)),
            prompt_length=4,
            max_context=12,
            max_new_tokens=4,
            prompt_position="start",
            prompt_offset=None,
        )
        self.assertEqual(prompt, [0, 1, 2, 3])
        self.assertEqual(meta["prompt_start"], 0)
        self.assertEqual(meta["prompt_end"], 4)

    def test_slice_prompt_middle(self):
        prompt, meta = slice_prompt(
            full_prompt_ids=list(range(20)),
            prompt_length=6,
            max_context=20,
            max_new_tokens=8,
            prompt_position="middle",
            prompt_offset=None,
        )
        self.assertEqual(prompt, [7, 8, 9, 10, 11, 12])
        self.assertEqual(meta["prompt_start"], 7)
        self.assertEqual(meta["prompt_end"], 13)

    def test_slice_prompt_offset_is_clamped(self):
        prompt, meta = slice_prompt(
            full_prompt_ids=list(range(10)),
            prompt_length=5,
            max_context=16,
            max_new_tokens=4,
            prompt_position="start",
            prompt_offset=99,
        )
        self.assertEqual(prompt, [5, 6, 7, 8, 9])
        self.assertEqual(meta["prompt_start"], 5)
        self.assertEqual(meta["prompt_end"], 10)

    def test_slice_prompt_rejects_invalid_lengths(self):
        with self.assertRaises(ValueError):
            slice_prompt(
                full_prompt_ids=[1, 2, 3],
                prompt_length=0,
                max_context=10,
                max_new_tokens=2,
                prompt_position="start",
                prompt_offset=None,
            )

        with self.assertRaises(ValueError):
            slice_prompt(
                full_prompt_ids=[1, 2, 3],
                prompt_length=2,
                max_context=4,
                max_new_tokens=4,
                prompt_position="start",
                prompt_offset=None,
            )

    def test_longest_run(self):
        self.assertEqual(longest_run([]), 0)
        self.assertEqual(longest_run([1, 1, 2, 2, 2, 3]), 3)

    def test_repeated_ngram_ratio(self):
        self.assertEqual(repeated_ngram_ratio([1, 2, 3], 4), 0.0)
        self.assertAlmostEqual(repeated_ngram_ratio([1, 2, 1, 2, 1, 2], 2), 0.6)


if __name__ == "__main__":
    unittest.main()
