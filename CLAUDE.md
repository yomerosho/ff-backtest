# ff-backtest

Fantasy-football start/sit system: evidence packet → LLM debate → verdict
(start/sit + confidence + floor/median/ceiling). Two modes off one engine —
**backtest** (replay a past week, graded against what happened) and **live**
(predict the upcoming week before it's played). Primary interface is a Streamlit
dashboard; CLIs remain for backtesting/experiments.

## Layout

Flat — every module sits in the repo root and imports its siblings top-level
(`from data import ...`). **Run every command from the repo root**; there is no
package, so running from elsewhere breaks imports. Don't add sub-packages
without rewriting the imports.

| File | Role |
|---|---|
| `dashboard.py` | Streamlit app: auto-targets current season's upcoming week; theme, models, refresh, bulk-add |
| `dashboard_core.py` | UI-free `build_view` (unified backtest+live resolution), `PlayerView`, `latest_played_week` |
| `data.py` | `load_weekly` / `load_schedule` (cached under `.data_cache/`), `game_env` (Vegas), `PointInTimeStore`, `PlayerContext` |
| `backtest.py` | replay loop, `Predictor` protocol, `Verdict`, `BaselinePredictor`, `MockDebatePredictor` |
| `scoring.py` | `PredictionRecord`, hit rate, MAE, Brier, calibration curve, ECE |
| `llm_predictor.py` | `LLMDebatePredictor` (`predict_full` returns verdict + case_for/against), prompt, cache, `fit_calibrator`, `FakeClient` |
| `predict_roster.py` | evidence assembly: `matchup_and_news` (DvP + Vegas), `build_context`, decision-card CLI |
| `run_backtest_enriched.py` | backtest over enriched evidence; `collect` with balanced per-position selection |
| `run_backtest.py` | original ablation CLI, reliability plot |
| `experiments.py` | accuracy-diagnostic driver (per-position, bias, calibration test, CSV) |

## Live vs backtest resolution

`dashboard_core.build_view` is unified: if the target week has a stat line it uses
it (backtest); otherwise the week is UPCOMING — current team from the player's
latest game, opponent from `game_env` (schedule). `game_env` keeps opponent/home
for every scheduled game with Vegas fields `None` until lines post. `load_weekly`
tolerates a not-yet-published season. Caveat: Week 1 of a new season uses last
season's team (offseason moves not reflected) and thin evidence.

## Measured accuracy (2024, n=504 balanced, Haiku)

Start/sit *direction* ≈ recent-average baseline (~0.70). Real wins are projection
MAE (6.8 vs 7.15) and bias (+0.2 vs +1.6 — recent-average over-projects in-form
players). Vegas + injuries give the best calibration (ECE 0.055) and hit-rate on
starts (0.712). A stronger model did NOT help — bottleneck is evidence, not
reasoning. Edge is largest on borderline players.

**Survivorship blind spot — read every backtest number with this in mind.** The
eval only contains players who PLAYED (they need a stat line to grade against), so
anyone ruled Out is silently excluded. Of 504 rows only 22 had an injury
designation and all were "Questionable" — the feature's biggest live win (benching
an Out player) is unmeasurable here and the aggregate numbers understate it.
Don't "fix" this by scoring Out players as 0; that invents outcomes. Judge injury
work by live behavior (e.g. McCaffrey Out wk2: START .78/18.2 → SIT .02/0.0).

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

## Data loading

`data.load_weekly(seasons)` is the single loader — every path goes through it
(`PointInTimeStore.from_nflverse`, `predict_roster`, `run_backtest_enriched`).
It reads nflverse parquet releases directly over HTTP; `nfl_data_py` is gone
(stale, stopped before 2025). Needs `pyarrow`. There's no local caching, so each
run re-downloads; if that becomes annoying, cache in `load_weekly`.

Two properties of the data worth knowing:

- **Postseason is included** (weeks 18-22). Default week ranges stop at 17, so
  it doesn't bite by default. Widen them and you're scoring playoff games, where
  defense-vs-position only covers surviving teams.
- **Every position is present** (linemen, DBs, K), mostly with 0.0 points rather
  than null, so they survive the `dropna`. Callers filter to WR/RB/TE/QB.

`from_nflverse`'s `pregame` table is still a PLACEHOLDER: `consensus_proj` is
just the prior week's points. The enriched path derives a real opponent-adjusted
projection instead, and is the better basis for conclusions.

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
