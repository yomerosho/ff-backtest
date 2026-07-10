"""Replay loop and predictor interface.

A Predictor takes a point-in-time PlayerContext and returns a Verdict. The
Backtester walks a past season week by week, hands each predictor ONLY the
context (never the outcome), then attaches the true result afterward for
scoring. Swap `MockDebatePredictor` for your real news+research+stats moderator
at the same interface and nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Protocol

import numpy as np

from data import OUTCOME_COL, PlayerContext, PointInTimeStore
from scoring import PredictionRecord


@dataclass
class Verdict:
    verdict: str            # "start" | "sit"
    confidence: float       # 0..1 == P(hit)
    proj_median: float
    proj_floor: float
    proj_ceiling: float


def _p_at_least(mean: float, sd: float, threshold: float) -> float:
    """P(X >= threshold) for X ~ Normal(mean, sd)."""
    sd = max(sd, 1e-6)
    z = (mean - threshold) / sd
    return 0.5 * (1 + erf(z / sqrt(2)))


class Predictor(Protocol):
    def predict(self, ctx: PlayerContext, start_threshold: float) -> Verdict: ...


class BaselinePredictor:
    """No agents, no research. Projection = average of recent games. This is the
    ablation control: if the debate system can't beat this, the LLM layer isn't
    earning its cost."""

    def __init__(self, window: int = 4):
        self.window = window

    def predict(self, ctx: PlayerContext, start_threshold: float) -> Verdict:
        recent = ctx.recent(self.window)
        if recent.empty:
            proj, sd = start_threshold, start_threshold * 0.5
        else:
            proj = float(recent.mean())
            sd = float(recent.std(ddof=0)) if len(recent) > 1 else max(proj * 0.4, 3.0)
        conf = _p_at_least(proj, sd, start_threshold)
        return Verdict(
            "start" if proj >= start_threshold else "sit",
            conf, proj, max(0.0, proj - sd), proj + sd,
        )


class MockDebatePredictor:
    """Stand-in for the real debate. It uses the pregame `consensus_proj` signal
    (which the baseline ignores) to represent 'the agents synthesized research
    into a better projection.' This lets you validate the harness and SEE what
    calibrated vs overconfident output looks like — swap it for the real
    moderator when ready.

    overconfidence: 0 == honest; >0 stretches confidence toward 0/1 so you can
    watch the calibration curve bend away from the diagonal.
    """

    def __init__(self, skill: float = 0.8, overconfidence: float = 0.0, seed: int = 0):
        self.skill = skill
        self.overconfidence = overconfidence
        self.rng = np.random.default_rng(seed)

    def predict(self, ctx: PlayerContext, start_threshold: float) -> Verdict:
        recent = ctx.recent(4)
        recent_mean = float(recent.mean()) if not recent.empty else start_threshold
        signal = ctx.matchup.get("consensus_proj", recent_mean)
        # Blend the better pregame signal with recent form, weighted by skill.
        proj = self.skill * signal + (1 - self.skill) * recent_mean
        sd = float(recent.std(ddof=0)) if len(recent) > 1 else max(proj * 0.4, 4.0)
        conf = _p_at_least(proj, sd, start_threshold)
        if self.overconfidence:
            # push probability away from 0.5 toward the extremes
            conf = 0.5 + (conf - 0.5) * (1 + self.overconfidence)
            conf = min(0.999, max(0.001, conf))
        return Verdict(
            "start" if proj >= start_threshold else "sit",
            conf, proj, max(0.0, proj - sd), proj + sd,
        )


class Backtester:
    def __init__(self, store: PointInTimeStore):
        self.store = store

    def run(
        self,
        predictor: Predictor,
        season: int,
        weeks: range,
        start_threshold: float = 12.0,
        positions: tuple[str, ...] = ("WR", "RB", "TE", "QB"),
        min_history: int = 3,
        max_players_per_week: int | None = None,
    ) -> list[PredictionRecord]:
        records: list[PredictionRecord] = []
        for wk in weeks:
            active = self.store.players_active_in(season, wk)
            active = active[active["position"].isin(positions)]
            if max_players_per_week:
                active = active.head(max_players_per_week)
            for _, p in active.iterrows():
                pid = p["player_id"]
                ctx = self.store.context_for(pid, season, wk)
                if len(ctx.history) < min_history:
                    continue                       # not enough history to judge
                actual = self.store.actual_points(pid, season, wk)
                if actual is None:
                    continue
                v = predictor.predict(ctx, start_threshold)
                baseline_proj = float(ctx.recent(4).mean()) if not ctx.recent(4).empty else start_threshold
                records.append(PredictionRecord(
                    player_id=pid, name=ctx.name, position=ctx.position,
                    season=season, week=wk,
                    verdict=v.verdict, confidence=v.confidence,
                    proj_median=v.proj_median, proj_floor=v.proj_floor,
                    proj_ceiling=v.proj_ceiling,
                    actual_points=actual, baseline_proj=baseline_proj,
                    start_threshold=start_threshold,
                ))
        return records
