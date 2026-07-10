"""Give it your roster, get a prediction per player for a given week.

    python predict_roster.py --season 2024 --week 10 "Justin Jefferson" "Bijan Robinson"

This is the practical front end to the system. For each player it:
  1. Builds a point-in-time context (only games BEFORE the target week).
  2. Enriches it with real matchup context — the upcoming opponent and how many
     fantasy points that defense allows to the player's position so far this
     season (defense-vs-position, computed from the same weekly data).
  3. Runs the LLM debate and prints a decision card: start/sit, confidence,
     projected floor/median/ceiling, and the case for and against.

The matchup enrichment is the accuracy lever the bare backtest was missing:
instead of "here are his recent points," the agent now sees "he's recent form X,
and he's facing a defense that gives up the 3rd-most points to WRs."

Note on live use: nflverse weekly data covers COMPLETED games. To predict a
truly upcoming week you need the current season loaded (weeks through last
Sunday). Point --season/--week at any week that exists in the data; the same
code serves a live week the moment that data is available.
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data import OUTCOME_COL, PlayerContext
from llm_predictor import (
    LLMDebatePredictor,
    build_evidence_packet,
    _hash,
    _parse_json,
)

NEEDED = ["player_id", "player_display_name", "position", "recent_team",
          "season", "week", "opponent_team", OUTCOME_COL,
          "targets", "receptions", "carries"]


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    w = nfl.import_weekly_data(seasons)
    keep = [c for c in NEEDED if c in w.columns]
    w = w[keep].rename(columns={"player_display_name": "name", "recent_team": "team"})
    return w.dropna(subset=[OUTCOME_COL]).reset_index(drop=True)


def defense_vs_position(weekly: pd.DataFrame, season: int, week: int) -> dict:
    """How many PPR points each defense allows to each position, using only
    games in `season` BEFORE `week`. Returns {(defense, position): (avg, rank)}
    where rank 1 = allows the most (best matchup for an offense)."""
    prior = weekly[(weekly.season == season) & (weekly.week < week)]
    if prior.empty:
        return {}
    # points a defense allowed = points scored by opposing players against it
    allowed = (prior.groupby(["opponent_team", "position"])[OUTCOME_COL]
               .mean().reset_index()
               .rename(columns={"opponent_team": "defense", OUTCOME_COL: "avg_allowed"}))
    out = {}
    for pos, grp in allowed.groupby("position"):
        grp = grp.sort_values("avg_allowed", ascending=False).reset_index(drop=True)
        for rank, row in grp.iterrows():
            out[(row["defense"], pos)] = (float(row["avg_allowed"]), int(rank) + 1,
                                          float(grp["avg_allowed"].mean()), len(grp))
    return out


def resolve_player(weekly: pd.DataFrame, season: int, week: int, query: str):
    """Find the player row for the target week by (case-insensitive) name."""
    target = weekly[(weekly.season == season) & (weekly.week == week)]
    exact = target[target.name.str.lower() == query.lower()]
    hit = exact if not exact.empty else target[target.name.str.lower().str.contains(query.lower())]
    if hit.empty:
        return None, f"no match for '{query}' in {season} week {week}"
    if len(hit) > 1:
        names = ", ".join(hit.name.unique())
        return None, f"'{query}' is ambiguous — did you mean: {names}?"
    return hit.iloc[0], None


def build_context(weekly: pd.DataFrame, prow: pd.Series, dvp: dict) -> PlayerContext:
    pid, season, week = prow["player_id"], int(prow["season"]), int(prow["week"])
    hist = weekly[(weekly.player_id == pid)
                  & ((weekly.season < season) | ((weekly.season == season) & (weekly.week < week)))]
    opp = prow["opponent_team"]
    pos = prow["position"]
    matchup = {"opponent": opp}
    news = []

    key = (opp, pos)
    if key in dvp:
        avg, rank, league_avg, n = dvp[key]
        ordinal = {1: "most", n: "fewest"}.get(rank, f"{rank}th-most")
        news.append({"text": f"Matchup: {opp} defense allows {avg:.1f} PPR/game to "
                             f"{pos}s this season ({ordinal} of {n}; league avg {league_avg:.1f})."})
        # opponent-adjusted projection off recent form
        recent = hist.tail(4)[OUTCOME_COL]
        if not recent.empty and league_avg > 0:
            base = float(recent.mean())
            matchup["consensus_proj"] = round(base * (avg / league_avg), 1)

    return PlayerContext(player_id=pid, name=prow["name"], position=pos,
                         season=season, week=week, history=hist,
                         matchup=matchup, news=news)


def card(pred: LLMDebatePredictor, ctx: PlayerContext, threshold: float) -> str:
    v = pred.predict(ctx, threshold)
    # recover the narrative (case for/against) from the cached raw response
    case_for = case_against = ""
    if pred.cache:
        raw = pred.cache.get(_hash(pred.model + "\n" + build_evidence_packet(ctx, threshold)))
        if raw:
            try:
                d = _parse_json(raw)
                case_for, case_against = d.get("case_for", ""), d.get("case_against", "")
            except Exception:
                pass
    verdict = v.verdict.upper()
    conf = f"{v.confidence*100:.0f}%"
    opp = ctx.matchup.get("opponent", "?")
    lines = [
        f"  {ctx.name:<22} {ctx.position:<3} vs {opp:<4}   {verdict:<5}  conf {conf}",
        f"      proj: floor {v.proj_floor:.1f}  |  median {v.proj_median:.1f}  |  ceiling {v.proj_ceiling:.1f}",
    ]
    if case_for:
        lines.append(f"      for : {case_for}")
    if case_against:
        lines.append(f"      risk: {case_against}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("players", nargs="+", help="player names, quoted")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="PPR points that count as a startable week")
    args = ap.parse_args()

    print(f"Loading data for {args.season}...")
    weekly = load_weekly(sorted({args.season - 1, args.season}))
    dvp = defense_vs_position(weekly, args.season, args.week)
    pred = LLMDebatePredictor(model=args.model)

    print(f"\nPredictions for {args.season} week {args.week} "
          f"(start threshold {args.threshold:.0f} PPR pts):\n")
    for query in args.players:
        prow, err = resolve_player(weekly, args.season, args.week, query)
        if err:
            print(f"  {query:<22} — {err}")
            continue
        ctx = build_context(weekly, prow, dvp)
        print(card(pred, ctx, args.threshold))
        print()


if __name__ == "__main__":
    main()
