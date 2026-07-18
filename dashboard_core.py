"""Assemble everything the dashboard shows for one player-week — separated from
the Streamlit UI so it can be tested without a browser.

The organizing idea is the BENCHMARK: recent-average (last 4 games), which is
exactly what the no-agents baseline uses. A player "outperforms his benchmark"
when the debate projects him meaningfully above that recent average AND explains
why with pregame facts the average can't see (matchup, game environment).

Resolution is UNIFIED for backtest and live: if the target week already has a
stat line (a completed/graded week) it uses it; otherwise it treats the week as
UPCOMING and pulls the player's current team from their latest game and the
opponent from the schedule — so you can predict a week before it is played.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from data import (OUTCOME_COL, PlayerContext, load_weekly, load_schedule,
                  game_env, load_injuries, injury_map)
from predict_roster import defense_vs_position, matchup_and_news
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
    injury_note: str = ""        # e.g. "Injury report: Questionable (Hamstring)…"
    injury_status: str = ""      # "Out" | "Doubtful" | "Questionable" | ""
    upcoming: bool = False       # target week not yet played (opponent from schedule)
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
    """Weekly frame + game-environment map + injury map for a season (and its
    predecessor)."""
    seasons = seasons_for(season)
    weekly = load_weekly(seasons)
    env = game_env(load_schedule(seasons))
    injuries = injury_map(load_injuries(seasons))
    return weekly, env, injuries


def latest_played_week(weekly: pd.DataFrame, season: int) -> int:
    """Highest week with stats in `season` (0 if the season hasn't started)."""
    s = weekly[weekly.season == season]
    return int(s["week"].max()) if not s.empty else 0


def _err(query, season, week, threshold, msg) -> PlayerView:
    return PlayerView(name=query, position="", team="", opponent="", season=season,
                      week=week, verdict="", confidence=0.0, proj_floor=0,
                      proj_median=0, proj_ceiling=0, benchmark=0, threshold=threshold,
                      case_for="", case_against="", error=msg)


def build_view(weekly: pd.DataFrame, env: dict, injuries: dict,
               pred: LLMDebatePredictor, query: str, season: int, week: int,
               threshold: float) -> PlayerView:
    season, week = int(season), int(week)

    # --- resolve the player's identity across ALL data (not just target week) --
    ql = query.strip().lower()
    exact = weekly[weekly["name"].str.lower() == ql]
    cand = exact if not exact.empty else \
        weekly[weekly["name"].str.lower().str.contains(re.escape(ql), na=False)]
    if cand.empty:
        return _err(query, season, week, threshold, f"no match for '{query}'")
    distinct = cand["name"].unique()
    if len(distinct) > 1:
        return _err(query, season, week, threshold,
                    f"'{query}' is ambiguous — did you mean: {', '.join(distinct[:5])}?")

    rows = weekly[weekly["player_id"] == cand.iloc[0]["player_id"]].sort_values(["season", "week"])
    latest = rows.iloc[-1]
    pid, name, position = latest["player_id"], latest["name"], latest["position"]

    # --- opponent: from the target week's stat line if played, else schedule ---
    target = rows[(rows["season"] == season) & (rows["week"] == week)]
    if not target.empty:
        team = target.iloc[0].get("team")
        opponent = target.iloc[0]["opponent_team"]
        upcoming = False
    else:
        team = latest.get("team")
        ge = env.get((team, season, week))
        if not ge:
            return _err(query, season, week, threshold,
                        f"{name} ({team}) has no game in {season} Week {week} — "
                        f"bye week, or the week isn't scheduled yet.")
        opponent = ge["opponent"]
        upcoming = True

    hist = rows[(rows["season"] < season) | ((rows["season"] == season) & (rows["week"] < week))]
    if hist.empty:
        return _err(query, season, week, threshold,
                    f"no games on record for {name} before {season} Week {week}.")

    dvp = defense_vs_position(weekly, season, week)
    injury = injuries.get((pid, season, week)) if injuries else None
    matchup, news = matchup_and_news(hist, team, opponent, position, dvp, env,
                                     season, week, injury)
    ctx = PlayerContext(player_id=pid, name=name, position=position, season=season,
                        week=week, history=hist, matchup=matchup, news=news)
    v, data = pred.predict_full(ctx, threshold)

    recent = hist.tail(6)
    recent_games = [{"week": int(r["week"]), "points": float(r[OUTCOME_COL])}
                    for _, r in recent.iterrows()]
    benchmark = float(ctx.recent(4).mean()) if not ctx.recent(4).empty else threshold
    notes = [n.get("text", "") for n in news]

    return PlayerView(
        name=name, position=position, team=str(team or ""), opponent=opponent,
        season=season, week=week, verdict=v.verdict, confidence=v.confidence,
        proj_floor=v.proj_floor, proj_median=v.proj_median, proj_ceiling=v.proj_ceiling,
        benchmark=benchmark, threshold=threshold,
        case_for=data.get("case_for", ""), case_against=data.get("case_against", ""),
        recent_games=recent_games,
        matchup_note=next((t for t in notes if t.startswith("Matchup")), ""),
        game_note=next((t for t in notes if t.startswith("Game environment")), ""),
        injury_note=next((t for t in notes if t.startswith("Injury report")), ""),
        injury_status=matchup.get("injury_status", ""),
        upcoming=upcoming,
    )
