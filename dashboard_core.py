"""Assemble everything the dashboard shows for one player-week — separated from
the Streamlit UI so it can be tested without a browser.

The organizing idea is the BENCHMARK: recent-average (last 4 games), which is
exactly what the no-agents baseline uses. A player "outperforms his benchmark"
when the debate projects him meaningfully above that recent average AND explains
why with pregame facts the average can't see (matchup, game environment).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data import OUTCOME_COL, PlayerContext, load_weekly, load_schedule, game_env
from predict_roster import defense_vs_position, build_context, resolve_player
from llm_predictor import LLMDebatePredictor


@dataclass
class PlayerView:
    name: str
    position: str
    team: str
    opponent: str
    season: int
    week: int
    verdict: str                 # "start" | "sit"
    confidence: float            # 0..1
    proj_floor: float
    proj_median: float
    proj_ceiling: float
    benchmark: float             # recent-4-game average (the baseline)
    threshold: float
    case_for: str
    case_against: str
    recent_games: list[dict] = field(default_factory=list)   # [{week, points, ...}]
    matchup_note: str = ""
    game_note: str = ""
    error: str = ""

    @property
    def edge(self) -> float:
        """How far the projection sits above the recent-average benchmark."""
        return self.proj_median - self.benchmark

    @property
    def beats_benchmark(self) -> bool:
        return self.proj_median > self.benchmark

    @property
    def clears_threshold(self) -> bool:
        return self.proj_median >= self.threshold


def seasons_for(season: int) -> list[int]:
    """Load the target season plus the prior one, for early-week history."""
    return sorted({season - 1, season})


def load_bundle(season: int):
    """Weekly frame + game-environment map for a season (and its predecessor)."""
    seasons = seasons_for(season)
    weekly = load_weekly(seasons)
    env = game_env(load_schedule(seasons))
    return weekly, env


def build_view(weekly: pd.DataFrame, env: dict, pred: LLMDebatePredictor,
               query: str, season: int, week: int, threshold: float) -> PlayerView:
    prow, err = resolve_player(weekly, season, week, query)
    if err:
        return PlayerView(name=query, position="", team="", opponent="",
                          season=season, week=week, verdict="", confidence=0.0,
                          proj_floor=0, proj_median=0, proj_ceiling=0,
                          benchmark=0, threshold=threshold, case_for="",
                          case_against="", error=err)

    dvp = defense_vs_position(weekly, season, week)
    ctx = build_context(weekly, prow, dvp, env)
    v, data = pred.predict_full(ctx, threshold)

    recent = ctx.history.tail(6)
    recent_games = [
        {"week": int(r["week"]), "points": float(r[OUTCOME_COL]),
         "targets": r.get("targets"), "carries": r.get("carries")}
        for _, r in recent.iterrows()
    ]
    benchmark = float(ctx.recent(4).mean()) if not ctx.recent(4).empty else threshold

    notes = [n.get("text", "") for n in ctx.news]
    matchup_note = next((t for t in notes if t.startswith("Matchup")), "")
    game_note = next((t for t in notes if t.startswith("Game environment")), "")

    return PlayerView(
        name=ctx.name, position=ctx.position, team=str(prow.get("team", "")),
        opponent=ctx.matchup.get("opponent", ""), season=season, week=week,
        verdict=v.verdict, confidence=v.confidence,
        proj_floor=v.proj_floor, proj_median=v.proj_median, proj_ceiling=v.proj_ceiling,
        benchmark=benchmark, threshold=threshold,
        case_for=data.get("case_for", ""), case_against=data.get("case_against", ""),
        recent_games=recent_games, matchup_note=matchup_note, game_note=game_note,
    )
