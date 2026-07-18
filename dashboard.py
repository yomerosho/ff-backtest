"""Start/sit dashboard — add players, see the case for each beating his benchmark.

    streamlit run dashboard.py

Needs ANTHROPIC_API_KEY (loaded from .env). Predictions are cached under
.llm_cache/, so re-checking a player is instant and free.
"""
from __future__ import annotations

import streamlit as st

from data import load_env
from llm_predictor import LLMDebatePredictor
from dashboard_core import load_bundle, build_view

load_env()

st.set_page_config(page_title="Start/Sit Debate", page_icon="🏈", layout="wide")

MODELS = {
    "Haiku (fast, cheap — best in testing)": "claude-haiku-4-5-20251001",
    "Sonnet (stronger, slower)": "claude-sonnet-5",
}

# Streamlit 1.59 exposes no overridable theme CSS variables, so switch the look
# by injecting CSS for the chosen palette. Accent colors are shared (they read
# fine on both backgrounds); only the surfaces/text change.
THEMES = {
    "Dark":  dict(bg="#0e1117", panel="#1a1d24", border="#2a2e39",
                  text="#e6e8eb", muted="#9aa0ab"),
    "Light": dict(bg="#ffffff", panel="#f6f8fa", border="#e2e5ea",
                  text="#1f2430", muted="#5b6270"),
}


def inject_theme(t: dict) -> None:
    st.markdown(f"""<style>
      .stApp {{ background-color:{t['bg']} !important; color:{t['text']} !important; }}
      [data-testid="stHeader"] {{ background:{t['bg']} !important; }}
      [data-testid="stSidebar"] {{ background-color:{t['panel']} !important; }}
      [data-testid="stSidebar"] * {{ color:{t['text']} !important; }}
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label,
      [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"],
      [data-testid="stMetricLabel"] {{ color:{t['text']} !important; }}
      [data-testid="stVerticalBlockBorderWrapper"] {{
          background-color:{t['panel']} !important; border-color:{t['border']} !important; }}
      [data-testid="stExpander"] details {{
          background-color:{t['panel']} !important; border-color:{t['border']} !important; }}
      .stTextInput input, .stNumberInput input {{
          background-color:{t['bg']} !important; color:{t['text']} !important;
          border-color:{t['border']} !important; }}
      .stApp table {{ color:{t['text']} !important; border-color:{t['border']} !important; }}
      .stApp thead th {{ color:{t['muted']} !important; }}
    </style>""", unsafe_allow_html=True)


# ---- sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    theme_name = st.radio("Theme", list(THEMES), horizontal=True, index=0)
    theme = THEMES[theme_name]
    inject_theme(theme)
    st.divider()
    season = st.number_input("Season", min_value=2015, max_value=2025, value=2024, step=1)
    week = st.number_input("Week", min_value=1, max_value=22, value=10, step=1)
    threshold = st.slider("Startable threshold (PPR pts)", 6.0, 24.0, 12.0, 0.5,
                          help="A week at or above this counts as a 'hit'.")
    model_label = st.radio("Model", list(MODELS), index=0)
    model = MODELS[model_label]
    st.caption("Data: nflverse weekly stats + Vegas lines. Only games *before* "
               "the selected week are used — no leakage.")

st.title("🏈 Start/Sit Debate")
st.markdown("Add players to see whether the debate expects them to **beat their "
            "recent-average benchmark** — and why.")

# ---- player list (session state) -------------------------------------------
if "players" not in st.session_state:
    st.session_state.players = ["Justin Jefferson", "Bijan Robinson"]


# header words and position/roster codes to ignore, so a two-column CSV like
# "Justin Jefferson,WR" contributes the name but not a bogus "WR" player.
_SKIP_TOKENS = {"player", "name", "players", "position", "pos", "team",
                "qb", "rb", "wr", "te", "k", "dst", "def", "flex"}


def parse_names(text: str) -> list[str]:
    """Names one-per-line or comma-separated (so a plain .txt, a single-column
    .csv, or a name,position .csv all work). Strips whitespace and surrounding
    quotes, drops header/position tokens, and de-dupes while preserving order."""
    names, seen = [], set()
    for line in (text or "").splitlines():
        for part in line.split(","):
            name = part.strip().strip('"').strip("'")
            low = name.lower()
            if not name or low in _SKIP_TOKENS:
                continue
            if low not in seen:
                seen.add(low)
                names.append(name)
    return names


def add_players(names: list[str]) -> int:
    existing = {p.lower() for p in st.session_state.players}
    added = 0
    for n in names:
        if n.lower() not in existing:
            st.session_state.players.append(n)
            existing.add(n.lower())
            added += 1
    return added


with st.form("add", clear_on_submit=True):
    c1, c2 = st.columns([4, 1])
    new = c1.text_input("Add a player", placeholder="e.g. Ja'Marr Chase",
                        label_visibility="collapsed")
    if c2.form_submit_button("Add", use_container_width=True) and new.strip():
        add_players([new.strip()])

with st.expander("Add several at once (paste a list or upload a file)"):
    with st.form("add_many", clear_on_submit=True):
        pasted = st.text_area(
            "Paste names — one per line, or comma-separated",
            placeholder="Justin Jefferson\nBijan Robinson\nJa'Marr Chase",
            height=120)
        uploaded = st.file_uploader("…or upload a .txt / .csv", type=["txt", "csv"])
        if st.form_submit_button("Add all"):
            text = pasted or ""
            if uploaded is not None:
                text += "\n" + uploaded.getvalue().decode("utf-8", errors="ignore")
            names = parse_names(text)
            n = add_players(names)
            if n:
                st.success(f"Added {n} player{'s' if n != 1 else ''}.")
            else:
                st.info("No new players found in that list.")
    st.caption("Each player runs one debate call (cached after the first). "
               "A large list is fine but the first run will take a moment.")

if not st.session_state.players:
    st.info("Add a player above to get started.")
    st.stop()

if len(st.session_state.players) > 1:
    if st.button(f"Clear all ({len(st.session_state.players)})"):
        st.session_state.players = []
        st.rerun()


# ---- data + predictor (cached) ---------------------------------------------
@st.cache_data(show_spinner="Loading season data…")
def _bundle(season: int):
    return load_bundle(season)


weekly, env = _bundle(int(season))
pred = LLMDebatePredictor(model=model)


def range_bar(v) -> str:
    """Horizontal floor→ceiling bar with median, threshold and benchmark marks."""
    hi = max(v.proj_ceiling, v.threshold, v.benchmark) * 1.08 or 1.0
    def pct(x): return max(0.0, min(100.0, 100.0 * x / hi))
    fl, md, cl = pct(v.proj_floor), pct(v.proj_median), pct(v.proj_ceiling)
    thr, bm = pct(v.threshold), pct(v.benchmark)
    band = "#22c55e" if v.verdict == "start" else "#9ca3af"
    return f"""
    <div style="position:relative;height:46px;margin:6px 0 2px;">
      <div style="position:absolute;top:18px;width:100%;height:10px;
                  background:rgba(128,128,128,.18);border-radius:5px;"></div>
      <div style="position:absolute;top:18px;left:{fl}%;width:{max(cl-fl,1)}%;height:10px;
                  background:{band};opacity:.55;border-radius:5px;"></div>
      <div title="median" style="position:absolute;top:12px;left:{md}%;width:3px;height:22px;
                  background:{band};transform:translateX(-1px);"></div>
      <div title="startable line" style="position:absolute;top:8px;left:{thr}%;width:2px;height:30px;
                  background:#ef4444;"></div>
      <div title="recent-average benchmark" style="position:absolute;top:8px;left:{bm}%;width:2px;
                  height:30px;background:#3b82f6;border-left:2px dashed #3b82f6;"></div>
    </div>
    <div style="font-size:.75rem;opacity:.7;">
      <span style="color:{band};">■ floor–ceiling</span> &nbsp;
      <span style="color:#ef4444;">┃ start line {v.threshold:.0f}</span> &nbsp;
      <span style="color:#3b82f6;">┋ benchmark {v.benchmark:.1f}</span>
    </div>"""


def render(v):
    if v.error:
        st.warning(f"**{v.name}** — {v.error}")
        return

    verdict = v.verdict.upper()
    color = "#16a34a" if v.verdict == "start" else "#6b7280"
    head, badge = st.columns([3, 1])
    head.subheader(f"{v.name}  ·  {v.position} — {v.team} vs {v.opponent}")
    badge.markdown(
        f"<div style='text-align:right;font-size:1.4rem;font-weight:700;color:{color};'>"
        f"{verdict}</div><div style='text-align:right;opacity:.7;'>"
        f"{v.confidence*100:.0f}% confidence</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Projection", f"{v.proj_median:.1f}")
    m2.metric("Benchmark", f"{v.benchmark:.1f}", help="Recent 4-game average")
    m3.metric("Edge vs benchmark", f"{v.edge:+.1f}",
              delta_color="normal" if v.edge >= 0 else "inverse")
    m4.metric("Range", f"{v.proj_floor:.0f}–{v.proj_ceiling:.0f}")

    st.markdown(range_bar(v), unsafe_allow_html=True)

    verdict_line = (
        f"Projects **{v.proj_median:.1f}**, "
        + ("**above**" if v.beats_benchmark else "**below**")
        + f" his {v.benchmark:.1f} recent-average benchmark and "
        + ("**above**" if v.clears_threshold else "**below**")
        + f" the {v.threshold:.0f}-pt start line."
    )
    st.markdown(verdict_line)
    if v.case_for:
        st.markdown(f"✅ **Case for:** {v.case_for}")
    if v.case_against:
        st.markdown(f"⚠️ **Risk:** {v.case_against}")

    with st.expander("Evidence the debate saw"):
        if v.matchup_note:
            st.markdown(f"- {v.matchup_note}")
        if v.game_note:
            st.markdown(f"- {v.game_note}")
        if v.recent_games:
            st.caption("Recent games (oldest → newest)")
            st.table([{"Week": g["week"], "PPR pts": round(g["points"], 1)}
                      for g in v.recent_games])


for name in list(st.session_state.players):
    with st.container(border=True):
        top = st.columns([12, 1])
        with top[0]:
            with st.spinner(f"Running the debate for {name}…"):
                view = build_view(weekly, env, pred, name, int(season),
                                  int(week), float(threshold))
            render(view)
        if top[1].button("✕", key=f"rm_{name}", help="Remove"):
            st.session_state.players.remove(name)
            st.rerun()

st.caption("Research tool, not betting advice. In backtests the debate's "
           "confidence ranking is informative — higher confidence, higher hit "
           "rate — but individual weeks carry large irreducible variance.")
