"""Accuracy experiment driver — richer than the CLI reports.

Runs the enriched (matchup + DvP) system against the baseline on a real season,
saves every prediction to CSV, and breaks the result down where the summary
hides it: projection bias, per-position accuracy, and a leakage-free calibration
test (fit the confidence->hit-rate map on early weeks, apply to held-out later
weeks). LLM responses are cached, so re-running after the first pass is free.

    python experiments.py --test-season 2024 --limit 15 --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import csv
import os

from data import (load_weekly, load_env, load_schedule, game_env,
                  load_injuries, injury_map)
from backtest import BaselinePredictor
from llm_predictor import LLMDebatePredictor, fit_calibrator
from run_backtest_enriched import collect
from scoring import (
    summarize, calibration_curve, expected_calibration_error, brier,
    hit_rate_on_starts, mae,
)

POSITIONS = ("WR", "RB", "TE", "QB")


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def save_csv(records, path: str) -> None:
    """floor/ceiling are saved so interval COVERAGE can be checked after the
    fact: they're meant to be ~10th/90th percentiles, so ~10% of actuals should
    land below the floor and ~10% above the ceiling."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "name", "position", "season", "week", "verdict",
                    "confidence", "proj_floor", "proj_median", "proj_ceiling",
                    "actual_points", "baseline_proj", "start_threshold", "hit"])
        for r in records:
            w.writerow([r.player_id, r.name, r.position, r.season, r.week, r.verdict,
                        f"{r.confidence:.4f}", f"{r.proj_floor:.2f}",
                        f"{r.proj_median:.2f}", f"{r.proj_ceiling:.2f}",
                        f"{r.actual_points:.2f}", f"{r.baseline_proj:.2f}",
                        r.start_threshold, int(r.hit)])


def projection_bias(records) -> float:
    """Mean signed error: >0 means the system projects HIGH on average."""
    xs = [r.proj_median - r.actual_points for r in records]
    return sum(xs) / len(xs) if xs else float("nan")


def per_position(records) -> dict:
    out = {}
    for pos in POSITIONS:
        rs = [r for r in records if r.position == pos]
        if not rs:
            continue
        out[pos] = dict(n=len(rs), hit_starts=hit_rate_on_starts(rs),
                        mae=mae(rs), bias=projection_bias(rs))
    return out


def calibration_experiment(records, split_frac: float = 0.6) -> dict:
    """Leakage-free: fit the calibrator on the earliest `split_frac` of weeks,
    measure ECE/Brier on the held-out later weeks, raw vs calibrated."""
    weeks = sorted({r.week for r in records})
    if len(weeks) < 4:
        return {}
    cut = weeks[int(len(weeks) * split_frac)]
    train = [r for r in records if r.week < cut]
    test = [r for r in records if r.week >= cut]
    if not train or not test:
        return {}

    calib = fit_calibrator(train)
    raw_ece, raw_brier = expected_calibration_error(test), brier(test)

    # apply the fitted map to the held-out confidences and re-measure
    import copy
    cal_test = []
    for r in test:
        rc = copy.copy(r)
        rc.confidence = min(1.0, max(0.0, calib(r.confidence)))
        cal_test.append(rc)
    cal_ece, cal_brier = expected_calibration_error(cal_test), brier(cal_test)

    return dict(cut_week=cut, n_train=len(train), n_test=len(test),
                raw_ece=raw_ece, cal_ece=cal_ece,
                raw_brier=raw_brier, cal_brier=cal_brier)


def report(name, sys_recs, base_recs, cal) -> None:
    s = summarize(sys_recs)
    b = summarize(base_recs)
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"  predictions ............. {s['n']}")
    print(f"  base hit rate ........... {_fmt(s['base_hit_rate'])}")
    print(f"  directional accuracy .... {_fmt(s['directional_accuracy'])}   (baseline {_fmt(b['directional_accuracy'])})")
    print(f"  hit rate on STARTs ...... {_fmt(s['hit_rate_on_starts'])}   (baseline {_fmt(b['hit_rate_on_starts'])})")
    print(f"  projection MAE .......... {_fmt(s['mae'])}   (baseline {_fmt(b['mae'])})")
    print(f"  projection bias ......... {_fmt(projection_bias(sys_recs))}   (baseline {_fmt(projection_bias(base_recs))})   [>0 = projects high]")
    print(f"  Brier ................... {_fmt(s['brier'])}   (base-rate {_fmt(s['baseline_brier'])})")
    print(f"  ECE ..................... {_fmt(s['ece'])}")

    print("\n  per-position (system):")
    print("    pos   n    hit@start   MAE     bias")
    for pos, d in per_position(sys_recs).items():
        print(f"    {pos:<4} {d['n']:<4} {_fmt(d['hit_starts']):<10} {_fmt(d['mae']):<7} {_fmt(d['bias'])}")

    if cal:
        print(f"\n  calibration test (fit on weeks <{cal['cut_week']}, "
              f"eval on >= {cal['cut_week']}; {cal['n_train']} train / {cal['n_test']} test):")
        print(f"    ECE   raw {_fmt(cal['raw_ece'])}  ->  calibrated {_fmt(cal['cal_ece'])}")
        print(f"    Brier raw {_fmt(cal['raw_brier'])}  ->  calibrated {_fmt(cal['cal_brier'])}")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, nargs="*", default=[2022, 2023])
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--weeks", type=int, nargs=2, default=[4, 17], metavar=("START", "END"))
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--tag", default="baseline", help="label for the saved CSV")
    ap.add_argument("--vegas", action="store_true",
                    help="add Vegas game-environment (implied total, spread) to the packet")
    ap.add_argument("--balanced", action="store_true",
                    help="evaluate a position-balanced, startable set (top-K/pos by form) "
                         "instead of head(limit) -- fixes RB under-sampling")
    ap.add_argument("--injuries", action="store_true",
                    help="add each player's pregame injury designation to the packet")
    args = ap.parse_args()

    # startable-pool sizes per week when --balanced; mirrors real roster depth
    per_position = {"WR": 12, "RB": 12, "TE": 6, "QB": 6} if args.balanced else None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: needs ANTHROPIC_API_KEY (set it in .env).")

    seasons = sorted(set(args.history + [args.test_season]))
    print(f"Loading nflverse weekly data for {seasons} ...")
    weekly = load_weekly(seasons)

    env = None
    if args.vegas:
        env = game_env(load_schedule(seasons))
        print(f"[vegas] game environment loaded for {len(env)} team-weeks")

    injuries = None
    if args.injuries:
        injuries = injury_map(load_injuries(seasons))
        print(f"[injuries] {len(injuries)} player-week designations loaded")

    system = LLMDebatePredictor(model=args.model)
    baseline = BaselinePredictor()
    weeks = range(args.weeks[0], args.weeks[1] + 1)
    print(f"[experiment] model={args.model}  weeks={args.weeks}  limit={args.limit}  "
          f"vegas={args.vegas}  injuries={args.injuries}  balanced={args.balanced}")

    sys_recs, base_recs = collect(system, baseline, weekly, args.test_season, weeks,
                                  args.threshold, POSITIONS, args.limit, env=env,
                                  injuries=injuries,
                                  per_position=per_position)

    out = f"records_{args.tag}_{args.test_season}.csv"
    save_csv(sys_recs, out)
    print(f"saved {len(sys_recs)} records -> {out}")

    cal = calibration_experiment(sys_recs)
    report(f"SYSTEM ({args.model}, enriched) — {args.test_season}", sys_recs, base_recs, cal)


if __name__ == "__main__":
    main()
