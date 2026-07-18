"""Point-in-time data access with strict week cutoffs.

The single most important job of this module is preventing *leakage*. When the
system is asked to predict week W, it may only see information that existed
BEFORE week W kicked off. Two kinds of data are handled differently:

  * outcomes  (fantasy_points_ppr) -> only rows STRICTLY BEFORE the target week
                are ever exposed to a predictor. The target week's outcome is
                fetched separately, by the scorer, after the fact.
  * pregame   (consensus projection, opponent, injury tag) -> the target week's
                row IS allowed, because it was known before kickoff.

If you ever find yourself putting the target week's `fantasy_points_ppr` into a
PlayerContext, that is the leakage bug that silently inflates accuracy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def load_env() -> None:
    """Load a local `.env` (e.g. ANTHROPIC_API_KEY) into the environment if
    python-dotenv is installed. Guarded so the no-key `--demo` path still runs on
    base requirements.txt; the live paths pull in python-dotenv via
    requirements-live.txt. A missing or empty `.env` is a silent no-op."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


OUTCOME_COL = "fantasy_points_ppr"

_PARQUET_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                "stats_player/stats_player_week_{}.parquet")

# On-disk cache so a run doesn't re-download (slow, and one network blip kills a
# long backtest). A COMPLETED season never changes; to refresh an in-progress
# season delete its file (or the whole dir). Override the location with
# FFB_DATA_CACHE. Gitignored.
_DATA_CACHE = Path(os.environ.get("FFB_DATA_CACHE", ".data_cache"))

# Columns any predictor might use. `targets`/`receptions`/`carries` are optional
# usage signals that the evidence packet includes when present.
WEEKLY_COLS = ["player_id", "player_display_name", "position", "team",
               "opponent_team", "season", "week", OUTCOME_COL,
               "targets", "receptions", "carries"]


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    """Read nflverse weekly player stats straight from the current parquet
    release. This is the ONE loader — `nfl_data_py` is stale and stops before
    2025, so going direct is what lets the harness see recent seasons at all.

    Rows are COMPLETED games, and include postseason (weeks 18-22). Fantasy
    leagues run on the regular season, so the default week ranges stop at 17;
    widen them and you are scoring playoff games, where a defense-vs-position
    table also only covers the teams still playing.
    """
    frames = []
    for s in seasons:
        cache = _DATA_CACHE / f"weekly_{s}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        try:
            w = pd.read_parquet(_PARQUET_URL.format(s))
        except Exception:
            # A not-yet-published season (e.g. an upcoming year before Week 1, or
            # the current week before its games post) has no parquet yet. Skip it
            # and use whatever prior seasons loaded -- do NOT cache the miss.
            continue
        keep = [c for c in WEEKLY_COLS if c in w.columns]
        w = w[keep].rename(columns={"player_display_name": "name"}).dropna(subset=[OUTCOME_COL])
        _DATA_CACHE.mkdir(parents=True, exist_ok=True)
        w.to_parquet(cache)
        frames.append(w)
    if not frames:
        raise RuntimeError(f"no weekly data available for any of {seasons}")
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


_SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def load_schedule(seasons: list[int]) -> pd.DataFrame:
    """Game schedule with Vegas lines (`spread_line`, `total_line`). These are
    set BEFORE kickoff, so they are leakage-safe pregame facts. nflverse
    convention: spread_line > 0 means the HOME team is favored by that many.

    Cached locally like the weekly data. The lines are static once set; delete
    the cache to pick up newly-scheduled games for an in-progress season."""
    cache = _DATA_CACHE / "games.parquet"
    if cache.exists():
        g = pd.read_parquet(cache)
    else:
        g = pd.read_csv(_SCHEDULE_URL)
        _DATA_CACHE.mkdir(parents=True, exist_ok=True)
        g.to_parquet(cache)
    return g[g["season"].isin(seasons)].reset_index(drop=True)


def game_env(schedule: pd.DataFrame) -> dict:
    """Map (team, season, week) -> pregame game environment. The implied team
    total = (game total +/- spread)/2 is the single number that captures 'how
    good a scoring spot is this' -- exactly what recent-average can't know."""
    out: dict = {}
    for _, r in schedule.iterrows():
        season, week = int(r["season"]), int(r["week"])
        home, away = r["home_team"], r["away_team"]
        total, spread = r.get("total_line"), r.get("spread_line")
        # Opponent/home are known as soon as the game is scheduled; the Vegas
        # fields stay None until the lines are posted (they arrive days before
        # kickoff). This lets an UPCOMING week resolve matchups before its lines
        # exist -- the whole point of live use.
        if pd.isna(total) or pd.isna(spread):
            out[(home, season, week)] = dict(implied_total=None, game_total=None,
                                             favored_by=None, is_home=True, opponent=away)
            out[(away, season, week)] = dict(implied_total=None, game_total=None,
                                             favored_by=None, is_home=False, opponent=home)
            continue
        total, spread = float(total), float(spread)
        # spread_line > 0 => home favored by that many. favored_by is each team's
        # own margin: positive = favored, negative = underdog.
        out[(home, season, week)] = dict(
            implied_total=round((total + spread) / 2.0, 1), game_total=total,
            favored_by=spread, is_home=True, opponent=away)
        out[(away, season, week)] = dict(
            implied_total=round((total - spread) / 2.0, 1), game_total=total,
            favored_by=-spread, is_home=False, opponent=home)
    return out


_INJURY_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
               "injuries/injuries_{}.parquet")


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    """Official weekly injury reports (per-season parquet). `report_status` is the
    game-status designation (Out / Doubtful / Questionable) filed BEFORE the game,
    so it is a leakage-safe pregame fact -- same category as the Vegas lines.
    Cached; a not-yet-published season is skipped rather than fatal."""
    frames = []
    for s in seasons:
        cache = _DATA_CACHE / f"injuries_{s}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        try:
            df = pd.read_parquet(_INJURY_URL.format(s))
        except Exception:
            continue
        _DATA_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "report_status",
                                     "report_primary_injury", "practice_status",
                                     "date_modified"])
    return pd.concat(frames, ignore_index=True)


def injury_map(injuries: pd.DataFrame) -> dict:
    """Map (player_id, season, week) -> the player's own pregame injury
    designation. Only players with a real `report_status` are included; everyone
    else is treated as healthy (no entry). gsis_id == the weekly player_id."""
    out: dict = {}
    if injuries.empty:
        return out
    inj = injuries.dropna(subset=["gsis_id"]).copy()
    # keep the final report per player-week (latest edit = pregame designation)
    if "date_modified" in inj.columns:
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(subset=["gsis_id", "season", "week"], keep="last")
    for _, r in inj.iterrows():
        status = r.get("report_status")
        if pd.isna(status) or not str(status).strip():
            continue
        out[(r["gsis_id"], int(r["season"]), int(r["week"]))] = dict(
            status=str(status).strip(),
            injury=(str(r["report_primary_injury"]).strip()
                    if pd.notna(r.get("report_primary_injury")) else ""),
            practice=(str(r["practice_status"]).strip()
                      if pd.notna(r.get("practice_status")) else ""),
        )
    return out


@dataclass
class PlayerContext:
    """Everything a predictor is allowed to see for one (player, week).

    Deliberately does NOT contain the target week's outcome.
    """

    player_id: str
    name: str
    position: str
    season: int
    week: int
    history: pd.DataFrame                     # weekly rows strictly before (season, week)
    news: list[dict] = field(default_factory=list)   # point-in-time news items
    matchup: dict = field(default_factory=dict)      # pregame context (proj, opponent, tag)

    def recent(self, n: int = 4) -> pd.Series:
        """Most recent n outcomes prior to the target week."""
        if self.history.empty:
            return pd.Series(dtype=float)
        return self.history[OUTCOME_COL].tail(n)


class PointInTimeStore:
    """Holds the full historical database. Discipline lives in what it *hands out*,
    not in what it stores — mirroring reality, where you own all past data but
    must not feed the future to the model."""

    def __init__(self, weekly: pd.DataFrame, pregame: pd.DataFrame):
        key = ["player_id", "season", "week"]
        self.weekly = weekly.sort_values(key).reset_index(drop=True)
        self.pregame = pregame.sort_values(key).reset_index(drop=True)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_nflverse(cls, seasons: list[int]) -> "PointInTimeStore":
        """Real data path, via `load_weekly`. Outcomes come from weekly data.

        The pregame table here is a PLACEHOLDER: `consensus_proj` is just the
        player's prior-week points, which is barely a projection. It exists so
        the plumbing runs end to end -- wire in a real consensus/ADP feed before
        trusting these numbers, or use the enriched path, which derives an
        opponent-adjusted projection from defense-vs-position instead.
        """
        weekly = load_weekly(seasons)

        # shift(1) means "previous row per player", so it is only the previous
        # WEEK if the rows are in chronological order first.
        pregame = weekly.sort_values(["player_id", "season", "week"]).copy()
        pregame["consensus_proj"] = (
            pregame.groupby("player_id")[OUTCOME_COL].shift(1)
        )
        pregame = pregame[["player_id", "season", "week", "consensus_proj"]]
        return cls(weekly, pregame)

    @classmethod
    def synthetic(cls, seasons: list[int], n_players: int = 40, seed: int = 7) -> "PointInTimeStore":
        """Self-contained fake data so the harness runs end-to-end with no
        network. Each player has a latent skill; a pregame `consensus_proj`
        signal is a noisy pre-kickoff estimate, and the actual points add more
        game-day noise on top. Because the signal is genuinely correlated with
        the outcome, a predictor that USES it (the mock debate) can beat one
        that only averages recent games (the baseline)."""
        rng = np.random.default_rng(seed)
        positions = ["WR", "RB", "TE", "QB"]
        rows, preg = [], []
        for i in range(n_players):
            pid = f"P{i:03d}"
            pos = positions[i % len(positions)]
            skill = rng.normal(12, 4)                      # latent mean fantasy points
            for season in seasons:
                for week in range(1, 18):
                    # pregame estimate known before kickoff
                    consensus = max(0.0, skill + rng.normal(0, 3))
                    # actual outcome = estimate + extra game-day variance
                    actual = max(0.0, consensus + rng.normal(0, 5))
                    rows.append(
                        dict(player_id=pid, name=f"Player {i}", position=pos,
                             season=season, week=week, **{OUTCOME_COL: round(actual, 1)})
                    )
                    preg.append(
                        dict(player_id=pid, season=season, week=week,
                             consensus_proj=round(consensus, 1))
                    )
        return cls(pd.DataFrame(rows), pd.DataFrame(preg))

    # ---- point-in-time access --------------------------------------------

    def _history_before(self, player_id: str, season: int, week: int) -> pd.DataFrame:
        d = self.weekly
        mask = (d.player_id == player_id) & (
            (d.season < season) | ((d.season == season) & (d.week < week))
        )
        return d[mask]

    def _pregame_row(self, player_id: str, season: int, week: int) -> dict:
        d = self.pregame
        row = d[(d.player_id == player_id) & (d.season == season) & (d.week == week)]
        return {} if row.empty else row.iloc[0].to_dict()

    def players_active_in(self, season: int, week: int) -> pd.DataFrame:
        d = self.weekly
        sub = d[(d.season == season) & (d.week == week)]
        return sub[["player_id", "name", "position"]].drop_duplicates()

    def context_for(self, player_id: str, season: int, week: int) -> PlayerContext:
        hist = self._history_before(player_id, season, week)
        name = hist["name"].iloc[-1] if not hist.empty else player_id
        pos = hist["position"].iloc[-1] if not hist.empty else "NA"
        pregame = self._pregame_row(player_id, season, week)
        matchup = {}
        if "consensus_proj" in pregame and pd.notna(pregame["consensus_proj"]):
            matchup["consensus_proj"] = float(pregame["consensus_proj"])
        return PlayerContext(
            player_id=player_id, name=name, position=pos,
            season=season, week=week, history=hist, matchup=matchup,
        )

    def actual_points(self, player_id: str, season: int, week: int):
        """Ground truth. For the SCORER only — never pass into a PlayerContext."""
        d = self.weekly
        row = d[(d.player_id == player_id) & (d.season == season) & (d.week == week)]
        if row.empty:
            return None
        return float(row[OUTCOME_COL].iloc[0])
