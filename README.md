# Fantasy football start/sit — debate + backtest

Give it a player and a week; it returns a **start/sit** call with a confidence, a
projected point range (floor / median / ceiling), and the **case for and against**
— then shows whether the player is expected to beat his own recent-average
benchmark, and why. It works two ways from the same engine:

- **Live** — predict the upcoming week before it's played (opponents from the
  schedule, stats through last week).
- **Backtest** — replay a past week and grade the call against what actually
  happened, so you can trust (or distrust) the system honestly.

All files live flat in one folder. Run every command from the repo root.

## Quickstart — the dashboard

```bash
pip install -r requirements.txt
cp .env.example .env                       # then paste your ANTHROPIC_API_KEY into it
python -m streamlit run dashboard.py
```

It opens on the **current season's upcoming week automatically** — you don't pick
a year. Add players (one at a time, or paste/upload a list), hit **Run debates**,
and read the cards. Each week during the season, click **🔄 Refresh data** to pull
the latest stats and Vegas lines. To replay a past week instead, open
*⚙️ Backtest a past week* in the sidebar and set the season/week.

No API key yet? The backtest harness runs a synthetic demo with zero setup:

```bash
pip install -r requirements.txt
python run_backtest.py --demo
```

## How it works

For a target (player, week) the engine assembles an **evidence packet** of facts
known *before kickoff*:

- **Recent form** — the player's last several games (this is also the *benchmark*:
  a recent-4-game average, i.e. what you'd guess with no analysis).
- **Matchup** — how many PPR points the upcoming opponent's defense allows to the
  player's position this season (defense-vs-position).
- **Game environment** — the Vegas implied team total and spread. This is the
  pregame signal recent-average structurally can't know, and the biggest
  accuracy lever the system has.

That packet goes to a single structured LLM call that argues the strongest case
for and against clearing the threshold, then synthesizes one verdict: a
probability, a point range, and the two one-line cases. A player "beats his
benchmark" when the debate projects him meaningfully above his recent average
*and* backs it with those pregame facts.

### No leakage — why the backtest is trustworthy

The one failure mode that matters is letting the system see the future, which
silently inflates accuracy until the tool looks brilliant and is worthless live.
Guards:

1. **Outcome cutoff.** A predictor only ever sees rows strictly *before* the
   target week. The target week's actual points go to the scorer, never into the
   evidence.
2. **Pregame-only facts.** Projections, matchup, and Vegas lines are known before
   kickoff and allowed; the outcome is not.
3. **Model memory.** A frontier LLM has read the internet through its training
   cutoff, so results on seasons it already knows can leak through the weights.
   Prefer testing on a season after the model's cutoff.

Live mode uses the identical resolution: if the target week has no box score yet,
it's treated as upcoming — current team from the player's latest game, opponent
from the schedule.

## Does it actually work? (measured)

Backtest on 2024, 504 balanced player-weeks, Haiku, vs a recent-average baseline:

| metric | system (+Vegas) | recent-average |
|---|---|---|
| directional accuracy | 0.70 | 0.69 |
| hit-rate on starts | 0.71 | 0.71 |
| projection MAE | **6.8** | 7.15 |
| projection bias | **+0.2** | +1.6 |
| calibration (held-out ECE) | **0.058** | — |

Read honestly:

- **The backtest understates the injury signal.** It can only grade players who
  *played*, so anyone ruled **Out** has no stat line and is excluded — of 504 eval
  rows only 22 carried a designation, all "Questionable." The biggest live win
  (benching an Out player) is structurally invisible here. On the Questionable
  rows it can see, injuries lifted directional accuracy 0.59 → 0.64. Live, the
  effect is far larger: Christian McCaffrey, Out in 2024 Week 2 on a 25.5
  recent-4 average, goes from START (conf .78, proj 18.2) to SIT (conf .02,
  proj 0.0) once the designation is in the packet.
- **On start/sit *direction* it's about even with recent-average** for clearly
  startable players — they mostly agree, so don't expect it to flip obvious calls.
- **It wins on projection quality.** Recent-average systematically *over-projects*
  in-form players by ~1.6 pts (they regress); the debate is nearly unbiased and
  ~0.35 pts lower error. Its edge is largest on **borderline** players, where
  recent form is misleading.
- **Vegas mainly improves calibration** — its confidence numbers are more
  trustworthy with lines in the packet.
- **A stronger model did not help** on this task — Haiku matched or beat Sonnet,
  because the bottleneck is evidence, not reasoning. Bigger models are available
  in the dashboard for comparison, but Haiku is the cheap, accurate default.

The confidence score is a usable dial: hit-rate rises with stated confidence, so
tighten your threshold when you can't afford a bust.

## Deploying to Streamlit Cloud

- **Main file path:** `dashboard.py`
- **Branch:** whichever branch actually contains the app — check it has
  `dashboard.py` before pointing the deploy at it.
- **Secrets:** `.env` is gitignored and won't exist on the server. Add the key
  under *Settings → Secrets* as `ANTHROPIC_API_KEY = "sk-ant-..."`; the app
  bridges Streamlit Secrets into the environment for you.
- **Dependencies** come from `requirements.txt` (the complete manifest).
- **⚠️ A public app spends your API key.** Streamlit Community Cloud apps are
  reachable by anyone with the URL, and every debate is a billed call on *your*
  key. Restrict viewers (or keep the URL private) before sharing, and set a spend
  limit in the Anthropic Console.
- The `.data_cache/` and `.llm_cache/` directories are ephemeral on Cloud — they
  rebuild after a restart, which costs a few extra calls but nothing breaks.

## Setup notes

- **Data** comes from nflverse (weekly parquet + schedule with Vegas lines),
  cached under `.data_cache/` so runs don't re-download. A completed season never
  changes; the dashboard's Refresh button (or deleting the cache) pulls fresh data
  for an in-progress season.
- **API key** loads from `.env` (`ANTHROPIC_API_KEY`). LLM responses are cached
  under `.llm_cache/` keyed by model + system prompt + evidence, so re-checking a
  player is instant and free; editing the prompt correctly re-asks.
- Both cache dirs and `.env` are gitignored.

## Command-line tools

```bash
# decision cards for specific players (any week that has data, or upcoming)
python predict_roster.py --season 2024 --week 10 "Justin Jefferson" "Bijan Robinson"

# full backtest with the enriched (matchup + Vegas) evidence
python run_backtest_enriched.py --history 2022 2023 --test-season 2024 --limit 15

# accuracy diagnostics: per-position, projection bias, calibration test, CSV dump
python experiments.py --test-season 2024 --balanced --vegas

# original ablation harness (mock or --live-llm), with a calibration plot
python run_backtest.py --demo
```

Keep `--limit` small on live-LLM backtests — without it, the harness calls the
API for every player in every week.

## Files

- `dashboard.py` / `dashboard_core.py` — the Streamlit app and its UI-free core
- `data.py` — point-in-time store, cutoff enforcement, nflverse + schedule loaders, `game_env`
- `llm_predictor.py` — the LLM moderator, prompt, response cache, calibrator
- `predict_roster.py` — evidence assembly (matchup + Vegas), decision cards, CLI
- `run_backtest_enriched.py` — backtest over the enriched evidence, balanced selection
- `run_backtest.py` — original ablation CLI, reliability plot
- `experiments.py` — accuracy-diagnostic driver
- `scoring.py` — hit rate, MAE, Brier, calibration curve, ECE
- `backtest.py` — replay loop, `Predictor` interface, baseline, mock debate
