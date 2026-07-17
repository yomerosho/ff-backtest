"""Run a walk-forward backtest and print an accuracy + calibration report.

Demo (no network, no API key, synthetic data):
    python run_backtest.py --demo

Real LLM predictor on synthetic data (needs ANTHROPIC_API_KEY; start small):
    python run_backtest.py --demo --live-llm --limit 5 --weeks 5 6

Real LLM predictor on real data (downloads nflverse parquet):
    python run_backtest.py --live-llm --history 2023 2024 --test-season 2025 --limit 10

The design mirrors the plan: restrict information to before the test season's
games, replay week by week, then score against what actually happened.
"""
from __future__ import annotations

import argparse
import os
import sys

from data import PointInTimeStore
from backtest import Backtester, BaselinePredictor, MockDebatePredictor
from scoring import summarize, calibration_curve


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


def maybe_plot(records, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    curve = calibration_curve(records, 10)
    xs = [b.mean_confidence for b in curve]
    ys = [b.observed_hit_rate for b in curve]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.plot(xs, ys, "o-", label="system")
    plt.xlabel("stated confidence")
    plt.ylabel("observed hit rate")
    plt.title("Reliability curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved calibration plot -> {path}")


def build_system(args):
    """Pick the predictor under test: the real LLM moderator or the mock."""
    if args.live_llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ERROR: --live-llm needs ANTHROPIC_API_KEY set in the environment.")
        from llm_predictor import LLMDebatePredictor
        print(f"[live-llm] model={args.model}  (responses cached under .llm_cache/)")
        return LLMDebatePredictor(model=args.model)
    return MockDebatePredictor(skill=0.8, overconfidence=args.overconfidence)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data, no network")
    ap.add_argument("--live-llm", action="store_true",
                    help="use the real LLM moderator instead of the mock (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--history", type=int, nargs="*", default=[2023, 2024],
                    help="seasons of history available before the test season")
    ap.add_argument("--test-season", type=int, default=2025)
    ap.add_argument("--weeks", type=int, nargs=2, default=[4, 17],
                    metavar=("START", "END"), help="inclusive week range to test")
    ap.add_argument("--limit", type=int, default=None,
                    help="max players per week — keep this SMALL for live-llm to bound cost")
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="PPR points that count as a startable week")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--overconfidence", type=float, default=0.0,
                    help="distort the MOCK's confidence to demo miscalibration")
    ap.add_argument("--plot", default="calibration.png")
    args = ap.parse_args()

    seasons = sorted(set(args.history + [args.test_season]))
    if args.demo:
        store = PointInTimeStore.synthetic(seasons)
        print(f"[demo] synthetic store built for seasons {seasons}")
    else:
        store = PointInTimeStore.from_nflverse(seasons)
        print(f"[live] loaded nflverse weekly data for seasons {seasons}")

    if args.live_llm and args.limit is None:
        print("WARNING: --live-llm without --limit will call the API for every "
              "player every week. Add e.g. --limit 5 --weeks 5 6 for a first run.")

    weeks = range(args.weeks[0], args.weeks[1] + 1)
    bt = Backtester(store)

    system = build_system(args)
    baseline = BaselinePredictor()

    sys_records = bt.run(system, args.test_season, weeks,
                         start_threshold=args.threshold, max_players_per_week=args.limit)
    base_records = bt.run(baseline, args.test_season, weeks,
                          start_threshold=args.threshold, max_players_per_week=args.limit)

    label = "SYSTEM (live LLM)" if args.live_llm else "SYSTEM (mock debate)"
    sys_sum = print_report(label, sys_records, args.bins)
    base_sum = print_report("BASELINE (recent average)", base_records, args.bins)

    print("\n=== ABLATION: does the debate beat the baseline? ===")
    mae_win = sys_sum["mae"] < base_sum["mae"]
    brier_win = sys_sum["brier"] < base_sum["brier"]
    print(f"  lower projection MAE?  {'YES' if mae_win else 'NO'}")
    print(f"  lower Brier score?     {'YES' if brier_win else 'NO'}")
    if not (mae_win or brier_win):
        print("  -> the agent layer is not earning its cost on this data.")

    maybe_plot(sys_records, args.plot)


if __name__ == "__main__":
    main()
