"""Backtest WITH matchup enrichment — the fair, at-scale test of whether
opponent / defense-vs-position context (plus the wider ranges) beats the bare
version you ran earlier.

Same walk-forward design as run_backtest.py, but every evidence packet now
carries the opponent and that defense's points-allowed-to-position, via the
exact enrichment the roster tool uses. Compare the SYSTEM numbers here against
your earlier bare-LLM backtest to see whether the enrichment moved the needle.

    python run_backtest_enriched.py --history 2022 2023 --test-season 2024 --limit 15 --weeks 4 17

Needs ANTHROPIC_API_KEY. Fresh calls unless cached, so keep --limit modest.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from predict_roster import defense_vs_position, build_context
from data import OUTCOME_COL, load_weekly, load_env
from backtest import BaselinePredictor
from llm_predictor import LLMDebatePredictor
from scoring import PredictionRecord, summarize, calibration_curve


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def print_report(name: str, records, n_bins: int) -> dict:
    s = summarize(records, n_bins=n_bins)
    print(f"\n=== {name} ===")
    print(f"  predictions ...................... {s['n']}")
    print(f"  base hit rate (all players) ...... {_fmt(s['base_hit_rate'])}")
    print(f"  start rate ....................... {_fmt(s['start_rate'])}")
    print(f"  directional accuracy ............. {_fmt(s['directional_accuracy'])}")
    print(f"  hit rate on STARTs ............... {_fmt(s['hit_rate_on_starts'])}"
          f"   (baseline {_fmt(s['baseline_hit_rate_on_starts'])})")
    print(f"  projection MAE ................... {_fmt(s['mae'])}"
          f"   (baseline {_fmt(s['baseline_mae'])})")
    print(f"  Brier score (lower better) ....... {_fmt(s['brier'])}"
          f"   (base-rate {_fmt(s['baseline_brier'])})")
    print(f"  expected calibration error ....... {_fmt(s['ece'])}")
    print("  calibration curve:")
    print("    conf-bin   mean_conf   observed   n")
    for b in s["calibration"]:
        print(f"    {b.lo:.1f}-{b.hi:.1f}      {b.mean_confidence:.2f}       "
              f"{b.observed_hit_rate:.2f}      {b.n}")
    return s


def _select_active(weekly, season, wk, positions, limit, per_position):
    """Who to evaluate in this week. Default (`limit`) takes the first N rows,
    which is arbitrary and starves low-frequency positions (RBs came out to ~1
    per week). `per_position` instead takes the top-K of each position by season-
    to-date average points -- the actually-rostered, startable players, and
    balanced across positions so per-position accuracy is measurable."""
    active = weekly[(weekly.season == season) & (weekly.week == wk)
                    & (weekly.position.isin(positions))]
    if not per_position:
        return active.head(limit) if limit else active
    prior = weekly[(weekly.season == season) & (weekly.week < wk)]
    form = prior.groupby("player_id")[OUTCOME_COL].mean()
    active = active.assign(_form=active["player_id"].map(form).fillna(0.0))
    parts = [active[active.position == pos].sort_values("_form", ascending=False)
             .head(per_position.get(pos, 0)) for pos in positions]
    return pd.concat(parts).drop(columns="_form")


def collect(system, baseline, weekly, season, weeks, threshold, positions, limit,
            min_history=3, env=None, per_position=None):
    """One pass over the season; builds enriched context per player-week and
    scores both predictors against the true outcome. `env` is an optional
    (team, season, week) -> game-environment map from `data.game_env`.
    `per_position` (e.g. {"WR":12,"RB":12,...}) selects a balanced, startable
    eval set instead of the arbitrary head(limit)."""
    sys_recs, base_recs = [], []
    for wk in weeks:
        dvp = defense_vs_position(weekly, season, wk)
        active = _select_active(weekly, season, wk, positions, limit, per_position)
        for _, prow in active.iterrows():
            ctx = build_context(weekly, prow, dvp, env)
            if len(ctx.history) < min_history:
                continue
            actual = float(prow[OUTCOME_COL])
            recent = ctx.recent(4)
            baseline_proj = float(recent.mean()) if not recent.empty else threshold
            for predictor, sink in ((system, sys_recs), (baseline, base_recs)):
                v = predictor.predict(ctx, threshold)
                sink.append(PredictionRecord(
                    player_id=ctx.player_id, name=ctx.name, position=ctx.position,
                    season=season, week=wk,
                    verdict=v.verdict, confidence=v.confidence,
                    proj_median=v.proj_median, proj_floor=v.proj_floor,
                    proj_ceiling=v.proj_ceiling,
                    actual_points=actual, baseline_proj=baseline_proj,
                    start_threshold=threshold,
                ))
    return sys_recs, base_recs


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, nargs="*", default=[2022, 2023])
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--weeks", type=int, nargs=2, default=[4, 17], metavar=("START", "END"))
    ap.add_argument("--limit", type=int, default=15,
                    help="max players per week — keep modest to bound cost")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: needs ANTHROPIC_API_KEY set in the environment.")

    seasons = sorted(set(args.history + [args.test_season]))
    print(f"Loading nflverse weekly data for {seasons} ...")
    weekly = load_weekly(seasons)

    system = LLMDebatePredictor(model=args.model)
    baseline = BaselinePredictor()
    weeks = range(args.weeks[0], args.weeks[1] + 1)

    print(f"[enriched] model={args.model}  (matchup + DvP in every packet; "
          f"cached under .llm_cache/)")
    sys_recs, base_recs = collect(system, baseline, weekly, args.test_season, weeks,
                                  args.threshold, ("WR", "RB", "TE", "QB"), args.limit)

    sys_sum = print_report("SYSTEM (enriched LLM)", sys_recs, args.bins)
    base_sum = print_report("BASELINE (recent average)", base_recs, args.bins)

    print("\n=== ABLATION: does the enriched debate beat the baseline? ===")
    mae_win = sys_sum["mae"] < base_sum["mae"]
    brier_win = sys_sum["brier"] < base_sum["brier"]
    print(f"  lower projection MAE?  {'YES' if mae_win else 'NO'}")
    print(f"  lower Brier score?     {'YES' if brier_win else 'NO'}")
    print("\nCompare SYSTEM above to your earlier BARE-LLM backtest to see if the "
          "matchup enrichment helped.")


if __name__ == "__main__":
    main()
