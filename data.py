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

from dataclasses import dataclass, field

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
    frames = [pd.read_parquet(_PARQUET_URL.format(s)) for s in seasons]
    w = pd.concat(frames, ignore_index=True)
    keep = [c for c in WEEKLY_COLS if c in w.columns]
    w = w[keep].rename(columns={"player_display_name": "name"})
    return w.dropna(subset=[OUTCOME_COL]).reset_index(drop=True)


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
