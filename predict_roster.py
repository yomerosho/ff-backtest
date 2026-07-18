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

Data comes from `data.load_weekly`, the same loader the backtests use, so what
you see here is what gets graded there.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from data import (OUTCOME_COL, PlayerContext, load_weekly, load_env, load_schedule,
                  game_env, load_injuries, injury_map)
from llm_predictor import LLMDebatePredictor


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
    hit = exact if not exact.empty else target[target.name.str.lower().str.contains(query.lower(), na=False)]
    if hit.empty:
        return None, f"no match for '{query}' in {season} week {week}"
    if len(hit) > 1:
        names = ", ".join(hit.name.unique())
        return None, f"'{query}' is ambiguous — did you mean: {names}?"
    return hit.iloc[0], None


def matchup_and_news(hist: pd.DataFrame, team, opponent, position: str,
                     dvp: dict, env: dict | None, season: int, week: int,
                     injury: dict | None = None):
    """Build the (matchup, news) evidence for one player-week from the defense-
    vs-position table, Vegas game environment, and the player's own injury
    designation. Shared by the backtest path (build_context) and the live
    dashboard so both see identical facts."""
    matchup = {"opponent": opponent}
    news: list[dict] = []

    # Injury status first — it's the strongest single signal (an Out player scores
    # ~0). Pregame designation, so leakage-safe.
    if injury and injury.get("status"):
        parts = [injury["status"]]
        if injury.get("injury"):
            parts.append(f"({injury['injury']})")
        if injury.get("practice"):
            parts.append(f"— {injury['practice']}")
        matchup["injury_status"] = injury["status"]
        news.append({"text": "Injury report: " + " ".join(parts) + "."})

    key = (opponent, position)
    if key in dvp:
        avg, rank, league_avg, n = dvp[key]
        ordinal = {1: "most", n: "fewest"}.get(rank, f"{rank}th-most")
        news.append({"text": f"Matchup: {opponent} defense allows {avg:.1f} PPR/game to "
                             f"{position}s this season ({ordinal} of {n}; league avg {league_avg:.1f})."})
        recent = hist.tail(4)[OUTCOME_COL]
        if not recent.empty and league_avg > 0:
            matchup["consensus_proj"] = round(float(recent.mean()) * (avg / league_avg), 1)

    # Vegas game environment — the pregame signal recent-average can't contain.
    # implied_total is None until the lines are posted, so guard on it.
    ge = env.get((team, season, week)) if env else None
    if ge and ge.get("implied_total") is not None:
        matchup["implied_total"] = ge["implied_total"]
        fav = ge["favored_by"]
        line = (f"favored by {fav:.1f}" if fav > 0
                else f"underdog by {abs(fav):.1f}" if fav < 0 else "pick'em")
        news.append({"text": f"Game environment: {team} implied for {ge['implied_total']:.1f} "
                             f"pts ({'home' if ge['is_home'] else 'away'} vs {ge['opponent']}, "
                             f"game total {ge['game_total']:.1f}, {line})."})
    return matchup, news


def build_context(weekly: pd.DataFrame, prow: pd.Series, dvp: dict,
                  env: dict | None = None, injuries: dict | None = None) -> PlayerContext:
    pid, season, week = prow["player_id"], int(prow["season"]), int(prow["week"])
    hist = weekly[(weekly.player_id == pid)
                  & ((weekly.season < season) | ((weekly.season == season) & (weekly.week < week)))]
    injury = injuries.get((pid, season, week)) if injuries else None
    matchup, news = matchup_and_news(hist, prow.get("team"), prow["opponent_team"],
                                     prow["position"], dvp, env, season, week, injury)
    return PlayerContext(player_id=pid, name=prow["name"], position=prow["position"],
                         season=season, week=week, history=hist,
                         matchup=matchup, news=news)


def card(pred: LLMDebatePredictor, ctx: PlayerContext, threshold: float) -> str:
    v, d = pred.predict_full(ctx, threshold)
    case_for, case_against = d.get("case_for", ""), d.get("case_against", "")
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
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("players", nargs="+", help="player names, quoted")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="PPR points that count as a startable week")
    args = ap.parse_args()

    print(f"Loading data for {args.season}...")
    seasons = sorted({args.season - 1, args.season})
    weekly = load_weekly(seasons)
    dvp = defense_vs_position(weekly, args.season, args.week)
    env = game_env(load_schedule(seasons))
    injuries = injury_map(load_injuries(seasons))
    pred = LLMDebatePredictor(model=args.model)

    print(f"\nPredictions for {args.season} week {args.week} "
          f"(start threshold {args.threshold:.0f} PPR pts):\n")
    for query in args.players:
        prow, err = resolve_player(weekly, args.season, args.week, query)
        if err:
            print(f"  {query:<22} — {err}")
            continue
        ctx = build_context(weekly, prow, dvp, env, injuries)
        print(card(pred, ctx, args.threshold))
        print()


if __name__ == "__main__":
    main()
