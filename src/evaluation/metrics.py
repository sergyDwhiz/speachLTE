"""
String distance metrics for ASR evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Classic dynamic programming edit distance."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


@dataclass
class ErrorRateResult:
    """Structured representation of an error-rate calculation."""

    distance: int
    reference_length: int

    @property
    def score(self) -> float:
        return self.distance / max(1, self.reference_length)


class WordErrorRate:
    """Compute word error rate over iterable transcripts."""

    def __call__(self, references: Iterable[str], hypotheses: Iterable[str]) -> ErrorRateResult:
        total_distance = 0
        total_words = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.split()
            hyp_words = hyp.split()
            total_distance += _levenshtein(ref_words, hyp_words)
            total_words += len(ref_words)
        return ErrorRateResult(distance=total_distance, reference_length=total_words)


class CharacterErrorRate:
    """Compute character error rate over iterable transcripts."""

    def __call__(self, references: Iterable[str], hypotheses: Iterable[str]) -> ErrorRateResult:
        total_distance = 0
        total_chars = 0
        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))
            total_distance += _levenshtein(ref_chars, hyp_chars)
            total_chars += len(ref_chars)
        return ErrorRateResult(distance=total_distance, reference_length=total_chars)
