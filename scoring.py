"""Scoring: grade a set of predictions against ground truth.

Three things are measured separately, because a start/sit hit rate alone hides
the two failures that matter most:

  * accuracy    - did "start" verdicts actually clear the threshold, and does
                  the system beat a dumb baseline?
  * calibration - when the system says 78% confidence, do ~78% of those hit?
                  This is the reliability curve. An uncalibrated confidence
                  number is decorative.
  * projection  - mean absolute error of the point projection vs actual, again
                  against a baseline projection.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionRecord:
    player_id: str
    name: str
    position: str
    season: int
    week: int
    verdict: str            # "start" | "sit"
    confidence: float       # 0..1  == P(player hits the start threshold)
    proj_median: float
    proj_floor: float
    proj_ceiling: float
    actual_points: float
    baseline_proj: float
    start_threshold: float

    @property
    def hit(self) -> bool:
        return self.actual_points >= self.start_threshold

    @property
    def said_start(self) -> bool:
        return self.verdict == "start"


def _safe_mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def directional_accuracy(records: list[PredictionRecord]) -> float:
    """Fraction where the start/sit call matched reality (start->hit, sit->miss)."""
    return _safe_mean(int(r.said_start == r.hit) for r in records)


def hit_rate_on_starts(records: list[PredictionRecord]) -> float:
    """Of the players it told you to START, how many actually cleared the bar."""
    starts = [r for r in records if r.said_start]
    return _safe_mean(int(r.hit) for r in starts)


def baseline_hit_rate_on_starts(records: list[PredictionRecord]) -> float:
    """Same, but for the baseline projection deciding start (proj >= threshold)."""
    starts = [r for r in records if r.baseline_proj >= r.start_threshold]
    return _safe_mean(int(r.hit) for r in starts)


def mae(records: list[PredictionRecord]) -> float:
    return _safe_mean(abs(r.proj_median - r.actual_points) for r in records)


def baseline_mae(records: list[PredictionRecord]) -> float:
    return _safe_mean(abs(r.baseline_proj - r.actual_points) for r in records)


def brier(records: list[PredictionRecord]) -> float:
    """Mean squared error of the confidence vs the binary hit outcome.
    Lower is better; 0.25 is what you get by always guessing 0.5."""
    return _safe_mean((r.confidence - int(r.hit)) ** 2 for r in records)


def baseline_brier(records: list[PredictionRecord]) -> float:
    """Brier if you always predicted the base rate (no skill reference point)."""
    base = _safe_mean(int(r.hit) for r in records)
    return _safe_mean((base - int(r.hit)) ** 2 for r in records)


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    mean_confidence: float
    observed_hit_rate: float
    n: int


def calibration_curve(records: list[PredictionRecord], n_bins: int = 10) -> list[CalibrationBin]:
    """Bin predictions by stated confidence; compare mean confidence to observed
    hit rate in each bin. A well-calibrated system hugs the diagonal
    (mean_confidence == observed_hit_rate)."""
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = [r for r in records if (lo <= r.confidence < hi) or (hi == 1.0 and r.confidence == 1.0)]
        if not in_bin:
            continue
        bins.append(CalibrationBin(
            lo=lo, hi=hi,
            mean_confidence=_safe_mean(r.confidence for r in in_bin),
            observed_hit_rate=_safe_mean(int(r.hit) for r in in_bin),
            n=len(in_bin),
        ))
    return bins


def expected_calibration_error(records: list[PredictionRecord], n_bins: int = 10) -> float:
    """Single-number calibration summary: weighted average gap between stated
    confidence and observed hit rate across bins. 0 == perfectly calibrated."""
    curve = calibration_curve(records, n_bins)
    total = sum(b.n for b in curve)
    if not total:
        return float("nan")
    return sum(b.n * abs(b.mean_confidence - b.observed_hit_rate) for b in curve) / total


def summarize(records: list[PredictionRecord], n_bins: int = 10) -> dict:
    return {
        "n": len(records),
        "start_rate": _safe_mean(int(r.said_start) for r in records),
        "base_hit_rate": _safe_mean(int(r.hit) for r in records),
        "directional_accuracy": directional_accuracy(records),
        "hit_rate_on_starts": hit_rate_on_starts(records),
        "baseline_hit_rate_on_starts": baseline_hit_rate_on_starts(records),
        "mae": mae(records),
        "baseline_mae": baseline_mae(records),
        "brier": brier(records),
        "baseline_brier": baseline_brier(records),
        "ece": expected_calibration_error(records, n_bins),
        "calibration": calibration_curve(records, n_bins),
    }
