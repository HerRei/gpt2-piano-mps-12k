from __future__ import annotations

import numpy as np


def build_remi_tokenizer():
    from miditok import REMI, TokenizerConfig

    config = TokenizerConfig(
        use_programs=False,
        one_token_stream_for_programs=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        num_velocities=32,
        beat_res={(0, 4): 8, (4, 12): 4},
    )
    return REMI(tokenizer_config=config)


def token_ids_array(tokenized) -> np.ndarray:
    if isinstance(tokenized, list):
        if not tokenized:
            raise ValueError("empty token list")
        if len(tokenized) != 1:
            raise ValueError(
                f"expected a single token stream for piano data, got {len(tokenized)} sequences"
            )
        tokenized = tokenized[0]

    ids = tokenized.ids if hasattr(tokenized, "ids") else tokenized
    return np.asarray(ids, dtype=np.int32)
