"""The real moderator predictor.

Reads a point-in-time evidence packet for one player-week, runs a start/sit
debate inside a single structured LLM call (strongest case for, strongest case
against, then a synthesized probability + point range), and returns a `Verdict`.

It plugs into the exact same interface the backtester already calls —
`predict(ctx, threshold) -> Verdict` — so it drops in wherever
`MockDebatePredictor` was, and every scoring/calibration metric keeps working.

Design choices that matter:
  * Evidence in, no invention. The agent sees only the facts assembled from the
    PlayerContext (recent stats, pregame projection, pregame news). The prompt
    forbids inventing stats — it argues over interpretation, not over what's true.
  * Caching. Every call is cached by a hash of (model + system prompt +
    evidence packet), so re-running a backtest costs nothing after the first
    pass. Iterate freely — editing the prompt invalidates the affected entries
    rather than silently serving you the old answers.
  * Calibration is a hook, not a hope. The raw LLM probability passes through an
    optional `calibrator` you fit from a prior backtest (see `fit_calibrator`).
    LLMs are overconfident out of the box; the harness measures it and this
    corrects it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from data import OUTCOME_COL, PlayerContext
from backtest import Verdict

MODEL_DEFAULT = "claude-sonnet-5"          # swap to claude-haiku-4-5-20251001 to cut cost

SYSTEM_PROMPT = """You are the moderator of a fantasy-football start/sit debate.

You receive an evidence packet of FACTS known before kickoff. Internally, build
the single strongest case that the player will EXCEED the point threshold, and
the single strongest case that he will FALL SHORT — using only the facts given.
Do not invent statistics, injuries, or matchups. Then synthesize one verdict.

Return ONLY a JSON object — no prose, no markdown fences — with these keys:
  "case_for":     one sentence, strongest reason he clears the threshold
  "case_against": one sentence, strongest reason he falls short
  "p_exceed":     number 0..1, probability he scores AT OR ABOVE the threshold
  "floor":        number, a realistic BAD game (~10th percentile outcome)
  "median":       number, the expected/typical outcome (~50th percentile)
  "ceiling":      number, a realistic BOOM game (~90th percentile outcome)

Fantasy scoring has a fat upper tail: high-usage skill players routinely double
their median in a boom week, and busts fall well below recent form. Make the
floor-to-ceiling range WIDE enough to reflect real week-to-week variance — a
narrow band around the median is almost always wrong. For a high-target WR or a
featured RB, the ceiling should be roughly twice the floor.

Calibration matters. p_exceed = 0.7 means that across many similar spots the
player clears the bar about 7 times in 10. Do not default to 0 or 1 — reflect
genuine uncertainty, and when the evidence is thin, stay near the base rate."""


def build_evidence_packet(ctx: PlayerContext, threshold: float) -> str:
    """Turn a point-in-time PlayerContext into the facts the agent may use.
    Contains nothing from the target week's outcome — that stays with the scorer."""
    lines = [
        f"Player: {ctx.name} ({ctx.position})",
        f"Upcoming game: season {ctx.season}, week {ctx.week}",
    ]
    if ctx.matchup.get("opponent"):
        lines.append(f"Opponent: {ctx.matchup['opponent']}")
    if "consensus_proj" in ctx.matchup:
        lines.append(f"Pregame consensus projection: {ctx.matchup['consensus_proj']:.1f} PPR pts")
    lines.append(f"Start threshold: {threshold:.1f} PPR points (>= this counts as a hit)")

    hist = ctx.history.tail(6)
    if hist.empty:
        lines.append("Recent games: none on record.")
    else:
        lines.append("Recent games (oldest to newest):")
        extra = [c for c in ("targets", "receptions", "carries", "snap_pct") if c in hist.columns]
        for _, r in hist.iterrows():
            parts = [f"wk{int(r['week'])}: {r[OUTCOME_COL]:.1f} pts"]
            for c in extra:
                if pd.notna(r.get(c)):
                    parts.append(f"{c}={r[c]:g}")
            lines.append("  " + ", ".join(parts))

    if ctx.news:
        lines.append("Pregame news:")
        for item in ctx.news:
            when = item.get("date") or item.get("timestamp") or ""
            what = item.get("text") or item.get("headline") or str(item)
            lines.append(f"  - {when} {what}".rstrip())
    return "\n".join(lines)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _parse_json(raw: str) -> dict:
    """Robustly pull a JSON object out of a model response."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1:
        s = s[i:j + 1]
    return json.loads(s)


class _FileCache:
    def __init__(self, path: str):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[str]:
        f = self.dir / f"{key}.json"
        return f.read_text() if f.exists() else None

    def set(self, key: str, value: str) -> None:
        (self.dir / f"{key}.json").write_text(value)


class LLMDebatePredictor:
    """Drop-in replacement for MockDebatePredictor, backed by a real LLM call."""

    def __init__(
        self,
        model: str = MODEL_DEFAULT,
        cache_dir: Optional[str] = ".llm_cache",
        client=None,
        calibrator: Optional[Callable[[float], float]] = None,
        max_tokens: int = 700,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.calibrator = calibrator or (lambda p: p)
        self.cache = _FileCache(cache_dir) if cache_dir else None
        self.system_prompt = system_prompt
        self._client = client            # inject a fake for offline tests

    def cache_key(self, packet: str) -> str:
        """Cache identity for one call. The system prompt belongs in here: a
        response is only reusable if the question AND the instructions that
        produced it are unchanged. Leave it out and a prompt edit reads back the
        old answers, making the change look like a no-op."""
        return _hash("\n".join((self.model, self.system_prompt, packet)))

    def _get_client(self):
        if self._client is None:
            import anthropic                       # imported only when going live
            self._client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        return self._client

    def _call_llm(self, packet: str) -> str:
        msg = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": packet}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def _to_verdict(self, data: dict, threshold: float) -> Verdict:
        p = min(1.0, max(0.0, float(data.get("p_exceed", 0.5))))
        p = min(1.0, max(0.0, float(self.calibrator(p))))
        median = float(data.get("median", threshold))
        floor = float(data.get("floor", max(0.0, median * 0.6)))
        ceiling = float(data.get("ceiling", median * 1.5))
        verdict = "start" if p > 0.5 else "sit"
        return Verdict(verdict, p, median, floor, ceiling)

    def predict(self, ctx: PlayerContext, start_threshold: float) -> Verdict:
        packet = build_evidence_packet(ctx, start_threshold)
        key = self.cache_key(packet)
        raw = self.cache.get(key) if self.cache else None
        if raw is None:
            raw = self._call_llm(packet)
            if self.cache:
                self.cache.set(key, raw)
        try:
            data = _parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            # A malformed response becomes a maximally-uncertain sit, not a crash.
            data = {"p_exceed": 0.5, "median": start_threshold}
        return self._to_verdict(data, start_threshold)


def fit_calibrator(records, n_bins: int = 10) -> Callable[[float], float]:
    """Fit a mapping stated-confidence -> observed-hit-rate from a prior
    backtest's records, so you can correct the LLM's overconfidence.

    Workflow: run a backtest (mock or raw LLM) -> fit_calibrator(records) ->
    pass the result as `calibrator=` -> re-run and watch ECE fall."""
    from scoring import calibration_curve

    curve = calibration_curve(records, n_bins)
    xs = [b.mean_confidence for b in curve]
    ys = [b.observed_hit_rate for b in curve]
    if len(xs) < 2:
        return lambda p: p

    def calibrate(p: float) -> float:
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for k in range(1, len(xs)):
            if p <= xs[k]:
                t = (p - xs[k - 1]) / (xs[k] - xs[k - 1] + 1e-9)
                return ys[k - 1] + t * (ys[k] - ys[k - 1])
        return ys[-1]

    return calibrate


class FakeClient:
    """Offline stand-in returning canned JSON, so the parse -> verdict path is
    testable with no API key. `responder(packet) -> str` lets tests vary output."""

    def __init__(self, responder: Optional[Callable[[str], str]] = None):
        self.messages = _FakeMessages(responder)


class _FakeMessages:
    def __init__(self, responder):
        self.responder = responder or (
            lambda packet: '{"case_for":"steady role","case_against":"tough matchup",'
                           '"p_exceed":0.64,"floor":7.5,"median":13.8,"ceiling":21.0}'
        )

    def create(self, model, max_tokens, system, messages):
        text = self.responder(messages[0]["content"])
        block = type("Block", (), {"type": "text", "text": text})()
        return type("Msg", (), {"content": [block]})()
