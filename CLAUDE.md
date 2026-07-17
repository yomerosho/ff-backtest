# ff-backtest

Walk-forward backtest harness for a fantasy-football start/sit system: evidence
packet → LLM debate → verdict (start/sit + confidence + floor/median/ceiling),
graded against what actually happened.

## Layout

Flat — every module sits in the repo root and imports its siblings top-level
(`from data import ...`). **Run every command from the repo root**; there is no
package, so running from elsewhere breaks imports. Don't add sub-packages
without rewriting the imports.

| File | Role |
|---|---|
| `data.py` | `PointInTimeStore` (cutoff enforcement), `PlayerContext`, synthetic + nflverse loaders |
| `backtest.py` | replay loop, `Predictor` protocol, `Verdict`, `BaselinePredictor`, `MockDebatePredictor` |
| `scoring.py` | `PredictionRecord`, hit rate, MAE, Brier, calibration curve, ECE |
| `llm_predictor.py` | `LLMDebatePredictor` (real Anthropic call), prompt, cache, `fit_calibrator`, `FakeClient` |
| `run_backtest.py` | main CLI: mock or bare-LLM, vs baseline, with ablation |
| `run_backtest_enriched.py` | same, but every packet carries opponent + defense-vs-position |
| `predict_roster.py` | practical front end: name your players, get decision cards |

## The prime directive: no leakage

The whole repo exists to answer "would this have been right, knowing only what
was knowable pre-kickoff." Leakage silently inflates accuracy until the system
looks brilliant and is worthless live. The invariant:

- **Outcomes** (`fantasy_points_ppr`) for the target week go to the *scorer only*.
- **Pregame facts** (projection, opponent, injury tag) for the target week are allowed.
- **Never put the target week's actual points into a `PlayerContext`.**

`PlayerContext.history` is strictly-before-target-week by construction. If you
touch context building, that's the property to preserve. A third leak is model
memory: a frontier LLM has read the internet through its training cutoff, so
testing on seasons it already knows leaks through the weights — prefer a test
season after the cutoff, and treat strong results on old seasons with suspicion.

## Predictor interface

Anything with `predict(ctx: PlayerContext, start_threshold: float) -> Verdict`
drops into the backtester unchanged and keeps every metric working. That's the
extension point — new predictors should implement it rather than modify the loop.

## Two data loaders that are NOT interchangeable

- `PointInTimeStore.from_nflverse()` (used by `run_backtest.py` without `--demo`)
  goes through `nfl_data_py`, which is stale and does **not** cover 2025+. Its
  `pregame` table is a placeholder: `consensus_proj` = simply the prior week's
  points, not a real projection feed.
- `predict_roster.load_weekly()` (used by `predict_roster.py` and
  `run_backtest_enriched.py`) reads nflverse parquet releases directly and does
  support 2025+.

So `run_backtest.py --test-season 2025` (the default!) will likely fail or come
back empty, while the enriched path handles it. Prefer the parquet loader for
new work. `pd.read_parquet` needs `pyarrow` — undeclared in requirements, but
pandas 3.x pulls it in anyway.

## LLM cost and the cache

Live calls are cached under `.llm_cache/` (gitignored) keyed by
`LLMDebatePredictor.cache_key()` = `sha256(model + system_prompt + packet)`, so
re-running a backtest is free after the first pass while a prompt edit correctly
misses and re-calls.

The prompt is part of that identity deliberately — a response is only reusable
if the question *and* the instructions that produced it are unchanged. Anything
reading the cache must go through `cache_key()` rather than recomputing the hash
(`predict_roster.card()` does this to recover case_for/case_against). If you
add another cache reader, use that method too.

Always bound cost with `--limit` on live runs; `run_backtest.py` warns when
`--live-llm` is passed without it. Note `--limit` uses `.head(n)`, i.e. the same
systematically-chosen players every week, not a random sample — fine for smoke
tests, biased for real conclusions.

## Testing without an API key

Two offline paths, both verified working:

- `python run_backtest.py --demo` — synthetic data, mock predictor, no network.
- `FakeClient` in `llm_predictor.py` — exercises the real parse → `Verdict` path
  with canned JSON: `LLMDebatePredictor(client=FakeClient(), cache_dir=None)`.

There is no test suite. If you add one, `FakeClient` is the seam to build on.

## Calibration is the point, not a nicety

An uncalibrated confidence number is decorative. LLMs are overconfident out of
the box; `fit_calibrator(records)` fits stated-confidence → observed-hit-rate
from a prior backtest and is passed back in as `calibrator=`. Watch ECE fall.
`--overconfidence` distorts the *mock* on purpose to show what a bent
reliability curve looks like.

## Conventions

- Module docstrings carry the design reasoning and are load-bearing — keep them
  current when behavior changes.
- Ablation before celebration: if the debate can't beat `BaselinePredictor`
  (recent average) on MAE and Brier, the LLM layer isn't earning its cost. Both
  runners print this verdict.
- Default model is `claude-sonnet-5`; `claude-haiku-4-5-20251001` cuts cost.
