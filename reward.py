"""Deterministic reward functions for the arithmetic GRPO experiment.

The model receives only a prompt. The GRPO trainer supplies the matching
reference answer from the training row to `exact_answer_reward` after sampling.
"""

from __future__ import annotations

import re


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def extract_tagged_answer(completion: str) -> str | None:
    """Extract an answer only when the completion contains exactly one valid tag.

    Extra prose or Markdown punctuation around a single tag is tolerated. This
    matches the baseline evaluator and keeps reward focused on mathematical
    correctness; malformed or repeated answer tags are rejected.
    """
    matches = ANSWER_TAG_PATTERN.findall(completion)
    if len(matches) != 1:
        return None
    return matches[0]


def exact_answer_reward(completion: str, reference_answer: str | int) -> float:
    """Return 1.0 for one correctly tagged integer answer, otherwise 0.0."""
    prediction = extract_tagged_answer(completion)
    return float(prediction is not None and prediction == str(reference_answer))
