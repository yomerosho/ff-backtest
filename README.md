# Fantasy football debate — backtest harness

A walk-forward backtest for the news + research + statistics → agent debate →
decision system. It answers one question honestly: *would this have been right
last season, if it had only known what was knowable at the time?*

All files live flat in one folder — no sub-packages to nest wrong.

## Quickstart

```bash
pip install -r requirements.txt
python run_backtest.py --demo
```

`--demo` uses synthetic data, so it runs with zero setup and no API keys. You'll
see a system-vs-baseline report and a calibration curve.

Real data (needs `nfl_data_py`):

```bash
python run_backtest.py --history 2023 2024 --test-season 2025
```

Watch what miscalibration looks like:

```bash
python run_backtest.py --demo --overconfidence 1.5
```

## The idea

Replay a past season week by week. For each player in each week, the predictor
sees **only** information that existed before kickoff — stats through the prior
week, the pregame projection, the injury report as it read that day. It produces
a verdict (start/sit) with a confidence and a projected point range. Then the
real result is revealed and scored.

## Why this is trustworthy (and where it breaks)

The one failure mode that matters is **leakage** — letting the system see the
future. It silently inflates accuracy until the thing looks brilliant and is
worthless live. Three guards:

1. **Outcome cutoff.** `PointInTimeStore` only hands a predictor rows strictly
   before the target week. The target week's actual points are fetched by the
   *scorer*, never placed in a `PlayerContext`.
2. **Point-in-time news.** Outcomes and *pregame* facts are separated. A
   projection or injury tag known before kickoff is allowed; a retrospective
   article that already knows the outcome is not.
3. **Model memory.** A frontier LLM has read the internet through its training
   cutoff, so testing on games it already knows leaks through the weights.
   Prefer a test season after the model's cutoff.

## What gets measured

- **Accuracy** — directional accuracy and hit-rate-on-starts, vs a
  recent-average baseline.
- **Calibration** — the reliability curve. When it says 78% confidence, do ~78%
  of those hit? Summarized as expected calibration error (ECE).
- **Projection error** — mean absolute error of the projection vs actual, vs a
  baseline projection.

The **ablation** at the end runs a no-agents baseline on the identical weeks. If
the debate can't beat recent-average, the LLM layer isn't earning its cost.

## Plugging in the real system

| Piece | Demo stand-in | Replace with |
|---|---|---|
| Data | `PointInTimeStore.synthetic()` | `PointInTimeStore.from_nflverse([...])` + a real pregame projection feed |
| Predictor | `MockDebatePredictor` | your moderator agent, exposing `predict(ctx, threshold) -> Verdict` |
| News | (none in demo) | a timestamped feed filtered to pregame; attach to `PlayerContext.news` |

## Files

- `data.py` — point-in-time store, cutoff enforcement, synthetic + nflverse loaders
- `backtest.py` — replay loop, `Predictor` interface, baseline, mock debate
- `scoring.py` — hit rate, MAE, Brier, calibration curve, ECE
- `run_backtest.py` — CLI, full-vs-baseline ablation, reliability plot

## Suggested .gitignore

```
__pycache__/
.venv/
*.png
```
