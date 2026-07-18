"""Start/sit dashboard — add players, see the case for each beating his benchmark.

    streamlit run dashboard.py          # main app file when deploying

Needs ANTHROPIC_API_KEY: locally from `.env`, on Streamlit Cloud from the app's
Secrets (`ANTHROPIC_API_KEY = "sk-ant-..."`). Predictions are cached under
.llm_cache/, so re-checking a player is instant and free.

Set APP_PASSWORD in Secrets to gate a publicly-deployed app: every debate is
billed to the owner's key, so the gate runs before any data load or API call.
"""
from __future__ import annotations

import datetime
import hmac
import os
import pathlib

import streamlit as st

from data import load_env
from llm_predictor import LLMDebatePredictor
from dashboard_core import load_bundle, build_view, latest_played_week

load_env()

# `.env` is gitignored, so a deployed app has none — bridge Streamlit's Secrets
# into the environment, which is where the anthropic SDK looks for the key.
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        _key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        _key = None
    if _key:
        os.environ["ANTHROPIC_API_KEY"] = str(_key)


def _configured(name: str):
    """Read a setting from the environment, falling back to Streamlit Secrets."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None


def password_gate() -> None:
    """Block the app behind a shared password when APP_PASSWORD is set.

    Community Cloud's free tier allows only one private app, so a public deploy
    needs its own gate: every debate is billed to the OWNER's API key. This runs
    before any data load or API call, so an unauthenticated visitor can never
    spend anything. Not real auth — one shared secret, no per-user accounts — but
    it does stop strangers and crawlers who find the URL.
    """
    expected = _configured("APP_PASSWORD")
    if not expected:
        st.warning("**This app is unlocked.** Anyone with the URL can run "
                   "debates billed to your API key. Set `APP_PASSWORD` in "
                   "Settings → Secrets before sharing it.", icon="⚠️")
        return
    if st.session_state.get("_authenticated"):
        return

    st.title("🏈 Start/Sit Debate")
    st.caption("Enter the app password to continue.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    if pw:
        # constant-time compare so a wrong guess leaks nothing via timing
        if hmac.compare_digest(str(pw), str(expected)):
            st.session_state["_authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def current_nfl_season() -> int:
    """The NFL season currently in play or next up. A season is labelled by the
    year it starts (Sept); Jan/Feb playoffs still belong to the prior year's
    season, so anything before March maps back a year."""
    t = datetime.date.today()
    return t.year if t.month >= 3 else t.year - 1


def auto_week(latest_played: int) -> int:
    """The week to predict by default: the next unplayed one during a live
    season, Week 1 before kickoff, or a normal week for a finished season."""
    if latest_played == 0:
        return 1
    if latest_played <= 17:
        return latest_played + 1
    return 10


@st.cache_data(show_spinner="Loading season data…")
def _bundle(season: int):
    return load_bundle(season)


def _refresh_data(season: int) -> None:
    """Drop cached data so the next load re-fetches fresh stats + Vegas lines —
    needed weekly during a live season. Removes this season's weekly file and the
    schedule, then clears Streamlit's in-memory cache."""
    for f in (pathlib.Path(".data_cache") / f"weekly_{season}.parquet",
              pathlib.Path(".data_cache") / "games.parquet"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    st.cache_data.clear()

st.set_page_config(page_title="Start/Sit Debate", page_icon="🏈", layout="wide")

password_gate()      # must precede any data load or API call

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "**No ANTHROPIC_API_KEY found.**\n\n"
        "- Running locally: put it in a `.env` file at the repo root.\n"
        "- Deployed on Streamlit Cloud: add it under **Settings → Secrets** as "
        "`ANTHROPIC_API_KEY = \"sk-ant-...\"`."
    )
    st.stop()

MODELS = {
    "Haiku 4.5 (fast, cheap — best in testing)": "claude-haiku-4-5-20251001",
    "Sonnet 5 (balanced)": "claude-sonnet-5",
    "Opus 4.8 (most capable)": "claude-opus-4-8",
    "Fable 5 (newest)": "claude-fable-5",
}

# Streamlit 1.59 exposes no overridable theme CSS variables, so switch the look
# by injecting CSS for the chosen palette. Accent colors are shared (they read
# fine on both backgrounds); only the surfaces/text change.
ACCENT = "#22c55e"  # shared green; used for button hover/borders in both themes
THEMES = {
    "Dark":  dict(bg="#0e1117", panel="#1a1d24", btn="#262b36", border="#2a2e39",
                  text="#e6e8eb", muted="#9aa0ab"),
    "Light": dict(bg="#ffffff", panel="#f6f8fa", btn="#ffffff", border="#d0d7de",
                  text="#1f2430", muted="#5b6270"),
}


def inject_theme(t: dict) -> None:
    st.markdown(f"""<style>
      .stApp {{ background-color:{t['bg']} !important; color:{t['text']} !important; }}
      [data-testid="stHeader"] {{ background:{t['bg']} !important; }}
      section[data-testid="stSidebar"] {{ background-color:{t['panel']} !important; }}
      section[data-testid="stSidebar"] * {{ color:{t['text']} !important; }}
      /* general text */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
      [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"],
      [data-testid="stMetricLabel"] {{ color:{t['text']} !important; }}
      /* cards + expander (both the body and the clickable header) */
      [data-testid="stVerticalBlockBorderWrapper"] {{
          background-color:{t['panel']} !important; border-color:{t['border']} !important; }}
      [data-testid="stExpander"] details,
      [data-testid="stExpander"] summary {{
          background-color:{t['panel']} !important; border-color:{t['border']} !important;
          color:{t['text']} !important; }}
      [data-testid="stExpander"] summary:hover {{ color:{ACCENT} !important; }}
      /* inputs: Streamlit wraps the real <input> in themed container divs that
         keep a light background, so theme the wrappers, not just the field */
      .stApp input, .stApp textarea {{
          background-color:{t['bg']} !important; color:{t['text']} !important;
          border-color:{t['border']} !important; }}
      [data-testid="stTextInputRootElement"],
      [data-testid="stTextAreaRootElement"],
      [data-testid="stNumberInputContainer"],
      [data-testid="stSelectbox"] div,
      .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
      .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {{
          background-color:{t['bg']} !important; border-color:{t['border']} !important; }}
      .stApp input::placeholder, .stApp textarea::placeholder {{ color:{t['muted']} !important; }}
      /* selectbox dropdown menu */
      [data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="option"] {{
          background-color:{t['panel']} !important; color:{t['text']} !important; }}
      /* file-uploader dropzone */
      [data-testid="stFileUploaderDropzone"] {{ background-color:{t['bg']} !important; }}
      [data-testid="stFileUploaderDropzone"] * {{ color:{t['text']} !important; }}
      /* buttons — explicit bg AND text so the label always contrasts */
      .stApp button {{
          background-color:{t['btn']} !important; color:{t['text']} !important;
          border:1px solid {t['border']} !important; }}
      .stApp button p, .stApp button span, .stApp button div {{ color:{t['text']} !important; }}
      .stApp button:hover {{ border-color:{ACCENT} !important; }}
      .stApp button:hover p, .stApp button:hover span {{ color:{ACCENT} !important; }}
      /* table */
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

    # By default the tool targets the current season's upcoming week — the live
    # user never picks a year. The manual pickers below are only for replaying a
    # specific past week (backtesting); they stay collapsed.
    this_season = current_nfl_season()
    with st.expander("⚙️ Backtest a past week", expanded=False):
        season = st.number_input("Season", min_value=2015, max_value=this_season,
                                 value=this_season, step=1)
        week_override = st.number_input("Week (0 = current / upcoming)",
                                        min_value=0, max_value=22, value=0, step=1)

    try:
        weekly, env, injuries = _bundle(int(season))
    except Exception as e:
        st.error(f"No data available for {int(season)} yet ({e}).")
        st.stop()
    lpw = latest_played_week(weekly, int(season))
    week = int(week_override) if week_override else auto_week(lpw)

    threshold = st.slider("Startable threshold (PPR pts)", 6.0, 24.0, 12.0, 0.5,
                          help="A week at or above this counts as a 'hit'.")
    model_label = st.selectbox("Model", list(MODELS), index=0)
    model = MODELS[model_label]
    if st.button("🔄 Refresh data", use_container_width=True,
                 help="Re-download this season's stats and the latest Vegas lines. "
                      "Use it each week during the live season."):
        _refresh_data(int(season))
        st.rerun()
    st.caption("Data: nflverse weekly stats + Vegas lines. Only games *before* "
               "the selected week are used — no leakage.")

st.title("🏈 Start/Sit Debate")
_upcoming = int(week) > lpw
_tag = "🟢 upcoming" if _upcoming else "backtest"
st.markdown(f"**Predicting {int(season)} · Week {int(week)}**  ·  {_tag}")
st.caption("Add players to see whether the debate expects them to beat their "
           "recent-average benchmark — and why. Each player is judged on his own "
           "prior games; the week above is set automatically to the next one to "
           "play (change it under *Backtest a past week*).")

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


# weekly + env were loaded in the sidebar (to auto-detect the upcoming week)
pred = LLMDebatePredictor(model=model)


def range_bar(v) -> str:
    """Horizontal floor→ceiling bar, with each value printed on it.

    Rows are stacked so labels can't collide: median above the bar, floor/ceiling
    just below their band ends, then the start line and benchmark below those.
    """
    hi = max(v.proj_ceiling, v.threshold, v.benchmark) * 1.12 or 1.0
    def pct(x): return max(0.0, min(100.0, 100.0 * x / hi))
    fl, md, cl = pct(v.proj_floor), pct(v.proj_median), pct(v.proj_ceiling)
    thr, bm = pct(v.threshold), pct(v.benchmark)
    band = "#22c55e" if v.verdict == "start" else "#9ca3af"
    lab = ("position:absolute;transform:translateX(-50%);white-space:nowrap;"
           "font-size:.72rem;line-height:1;")
    return f"""
    <div style="position:relative;height:80px;margin:8px 0 2px;">
      <!-- median, above the bar -->
      <div style="{lab}top:0;left:{md}%;font-weight:700;color:{band};">{v.proj_median:.1f}</div>
      <!-- track + floor..ceiling band -->
      <div style="position:absolute;top:20px;width:100%;height:12px;
                  background:rgba(128,128,128,.18);border-radius:6px;"></div>
      <div style="position:absolute;top:20px;left:{fl}%;width:{max(cl-fl,1)}%;height:12px;
                  background:{band};opacity:.5;border-radius:6px;"></div>
      <!-- ticks -->
      <div style="position:absolute;top:15px;left:{md}%;width:3px;height:22px;
                  background:{band};transform:translateX(-1.5px);"></div>
      <div style="position:absolute;top:15px;left:{thr}%;width:2px;height:22px;
                  background:#ef4444;transform:translateX(-1px);"></div>
      <div style="position:absolute;top:15px;left:{bm}%;width:2px;height:22px;
                  background:#3b82f6;transform:translateX(-1px);"></div>
      <!-- floor / ceiling values at the band ends -->
      <div style="{lab}top:40px;left:{fl}%;color:{band};">{v.proj_floor:.1f}</div>
      <div style="{lab}top:40px;left:{cl}%;color:{band};">{v.proj_ceiling:.1f}</div>
      <!-- reference lines -->
      <div style="{lab}top:58px;left:{thr}%;color:#ef4444;">start {v.threshold:.0f}</div>
      <div style="{lab}top:58px;left:{bm}%;color:#3b82f6;">avg {v.benchmark:.1f}</div>
    </div>
    <div style="font-size:.72rem;opacity:.65;">
      floor {v.proj_floor:.1f} · median {v.proj_median:.1f} · ceiling
      {v.proj_ceiling:.1f} — floor/ceiling are the model's rough 10th/90th
      percentile guesses, <b>not</b> bounds: about 1 game in 10 lands outside each end.
    </div>"""


def render(v):
    if v.error:
        st.warning(f"**{v.name}** — {v.error}")
        return

    verdict = v.verdict.upper()
    color = "#16a34a" if v.verdict == "start" else "#6b7280"
    head, badge = st.columns([3, 1])
    tag = "  ·  🟢 upcoming" if v.upcoming else ""
    head.subheader(f"{v.name}  ·  {v.position} — {v.team} vs {v.opponent}{tag}")
    badge.markdown(
        f"<div style='text-align:right;font-size:1.4rem;font-weight:700;color:{color};'>"
        f"{verdict}</div><div style='text-align:right;opacity:.7;'>"
        f"{v.confidence*100:.0f}% confidence</div>", unsafe_allow_html=True)

    if v.injury_status:
        ic = {"Out": "#ef4444", "Doubtful": "#f97316"}.get(v.injury_status, "#eab308")
        st.markdown(
            f"<div style='background:{ic}22;border-left:3px solid {ic};padding:.4rem .7rem;"
            f"border-radius:4px;margin:.2rem 0;'>🩹 <b>{v.injury_note[14:].strip() or v.injury_status}</b>"
            f"</div>", unsafe_allow_html=True)

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
        if v.injury_note:
            st.markdown(f"- {v.injury_note}")
        if v.matchup_note:
            st.markdown(f"- {v.matchup_note}")
        if v.game_note:
            st.markdown(f"- {v.game_note}")
        if v.recent_games:
            st.caption("Recent games (oldest → newest)")
            st.table([{"Week": g["week"], "PPR pts": round(g["points"], 1)}
                      for g in v.recent_games])


# ---- run debates on demand (not on every rerun) ----------------------------
# Views are cached in session_state keyed by (player, season, week, threshold,
# model), so adding players or toggling the theme does NOT fire API calls —
# debates run only when Refresh is pressed. Changing any setting makes the keys
# miss, so those players show as pending until the next run.
views = st.session_state.setdefault("views", {})


def view_key(name: str) -> tuple:
    return (name.lower(), int(season), int(week), float(threshold), model)


players = st.session_state.players
pending = [n for n in players if view_key(n) not in views]

c_run, c_clear = st.columns([3, 1])
run = c_run.button(
    f"🔄 Run debates ({len(pending)} pending)" if pending else "🔄 Refresh debates",
    type="primary", use_container_width=True,
    help="Runs the debate for each player at the current settings. Cached after "
         "the first run; changing model/week/threshold requires a refresh.")
if c_clear.button(f"Clear all ({len(players)})", use_container_width=True):
    st.session_state.players = []
    st.rerun()

if run:
    todo = pending or players            # nothing pending -> force a full refresh
    prog = st.progress(0.0, text="Running debates…")
    for i, n in enumerate(todo):
        prog.progress(i / len(todo), text=f"Debating {n} ({i+1}/{len(todo)})…")
        views[view_key(n)] = build_view(weekly, env, injuries, pred, n, int(season),
                                        int(week), float(threshold))
    prog.empty()
    pending = [n for n in players if view_key(n) not in views]

if pending:
    st.info(f"{len(pending)} player(s) awaiting a run at these settings — "
            f"hit **Run debates** above.")

for name in list(players):
    with st.container(border=True):
        top = st.columns([12, 1])
        with top[0]:
            v = views.get(view_key(name))
            if v is not None:
                render(v)
            else:
                st.markdown(f"**{name}** — _pending; hit Run debates._")
        if top[1].button("✕", key=f"rm_{name}", help="Remove"):
            st.session_state.players.remove(name)
            st.rerun()

st.caption("Research tool, not betting advice. In backtests the debate's "
           "confidence ranking is informative — higher confidence, higher hit "
           "rate — but individual weeks carry large irreducible variance.")
