# Fantasy football debate — backtest harness

A walk-forward backtest for the news + research + statistics → agent debate →
decision system. It answers one question honestly: *would this have been right
last season, if it had only known what was knowable at the time?*

All files live flat in one folder — no sub-packages to nest wrong. Run every
command from the repo root.

## Quickstart

```bash
pip install -r requirements.txt
python run_backtest.py --demo
```

`--demo` uses synthetic data, so it runs with zero setup and no API keys. You'll
see a system-vs-baseline report and a calibration curve.

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

## Going live

The real predictor runs the start/sit debate inside a single structured LLM call
— strongest case for, strongest case against, then a synthesized probability and
point range. It implements the same `predict(ctx, threshold) -> Verdict`
interface as the mock, so it drops in and every metric keeps working.

```bash
pip install -r requirements-live.txt
export ANTHROPIC_API_KEY=...

# smoke test: real LLM, synthetic data, tiny sample
python run_backtest.py --demo --live-llm --limit 5 --weeks 5 6
```

Responses are cached under `.llm_cache/` by model + system prompt + evidence, so
re-running a backtest is free after the first pass — while editing the prompt
correctly misses the cache and re-asks, instead of grading yesterday's answers.
**Keep `--limit` small on live runs** — without it, the harness calls the API for
every player in every week.

### Predict your actual roster

```bash
python predict_roster.py --season 2024 --week 10 "Justin Jefferson" "Bijan Robinson"
```

Prints a decision card per player: start/sit, confidence, floor/median/ceiling,
and the case for and against. Each context is enriched with the real upcoming
opponent and how many fantasy points that defense allows to the player's
position so far this season (defense-vs-position).

### Backtest with that enrichment

```bash
python run_backtest_enriched.py --history 2022 2023 --test-season 2024 --limit 15
```

Same walk-forward design, but every evidence packet carries the matchup context
— the at-scale test of whether the enrichment actually beats the bare version.

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
  of those hit? Summarized as expected calibration error (ECE). LLMs are
  overconfident out of the box; `fit_calibrator()` in `llm_predictor.py` fits a
  correction from a prior backtest's records.
- **Projection error** — mean absolute error of the projection vs actual, vs a
  baseline projection.

The **ablation** at the end runs a no-agents baseline on the identical weeks. If
the debate can't beat recent-average, the LLM layer isn't earning its cost.

## Data sources

Two loaders, and they are not interchangeable:

- `PointInTimeStore.from_nflverse()` — used by `run_backtest.py` without
  `--demo`. Goes through `nfl_data_py`, which is stale and does **not** cover
  2025+. Its pregame projection is a placeholder (prior week's points); wire in a
  real consensus feed for production use.
- `predict_roster.load_weekly()` — used by `predict_roster.py` and
  `run_backtest_enriched.py`. Reads the nflverse parquet releases directly and
  supports 2025+. Prefer this for new work.

## Files

- `data.py` — point-in-time store, cutoff enforcement, synthetic + nflverse loaders
- `backtest.py` — replay loop, `Predictor` interface, baseline, mock debate
- `scoring.py` — hit rate, MAE, Brier, calibration curve, ECE
- `llm_predictor.py` — the real LLM moderator, prompt, response cache, calibrator
- `run_backtest.py` — CLI, full-vs-baseline ablation, reliability plot
- `run_backtest_enriched.py` — backtest with matchup / defense-vs-position context
- `predict_roster.py` — roster front end, decision cards
