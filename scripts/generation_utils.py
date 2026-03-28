from typing import Dict, List, Optional, Tuple
import random


def parse_int_list(spec: Optional[str]) -> List[int]:
    if spec is None:
        return []
    items = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            items.append(int(chunk))
    return items


def parse_str_list(spec: Optional[str]) -> List[str]:
    if spec is None:
        return []
    items = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            items.append(chunk)
    return items


def slice_prompt(
    full_prompt_ids: List[int],
    prompt_length: int,
    max_context: int,
    max_new_tokens: int,
    prompt_position: str,
    prompt_offset: Optional[int],
) -> Tuple[List[int], Dict[str, int]]:
    if prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")

    allowed_prompt = max_context - max_new_tokens
    if allowed_prompt < 1:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} is too large for context window {max_context}"
        )

    final_prompt_len = min(len(full_prompt_ids), prompt_length, allowed_prompt)
    if final_prompt_len < 1:
        raise ValueError("prompt is empty after trimming")

    max_start = max(0, len(full_prompt_ids) - final_prompt_len)

    if prompt_offset is not None:
        start = min(max(prompt_offset, 0), max_start)
    elif prompt_position == "start":
        start = 0
    elif prompt_position == "middle":
        start = max_start // 2
    elif prompt_position == "random":
        start = random.randint(0, max_start) if max_start > 0 else 0
    else:
        raise ValueError(f"unsupported prompt position: {prompt_position}")

    end = start + final_prompt_len
    return full_prompt_ids[start:end], {
        "prompt_start": int(start),
        "prompt_end": int(end),
        "prompt_position": prompt_position,
        "prompt_total_source_tokens": int(len(full_prompt_ids)),
    }


def longest_run(token_ids: List[int]) -> int:
    if not token_ids:
        return 0
    longest = 1
    current = 1
    for prev, cur in zip(token_ids, token_ids[1:]):
        if cur == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def repeated_ngram_ratio(token_ids: List[int], n: int) -> float:
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def target_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return clamp01(1.0 - abs(value - target) / tolerance)
