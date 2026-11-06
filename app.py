import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="NFL Simulator (EPA + Usage)", layout="wide")
st.title("NFL Game Simulator — EPA + Player Usage")

@st.cache_data(ttl=3600)
def load_data():
    seasons = [2024, 2025, 2026]
    pbp = nfl.load_pbp(seasons).to_pandas()
    player_stats = nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()
    return pbp, player_stats, snaps, schedules

try:
    pbp, player_stats, snaps, schedules = load_data()
    st.sidebar.success("Data loaded")
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# --- Team EPA Ratings ---
@st.cache_data
def build_team_epa(pbp):
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    off = plays.groupby("posteam")["epa"].mean().rename("off_epa")
    def_ = plays.groupby("defteam")["epa"].mean().rename("def_epa")
    ratings = pd.concat([off, def_], axis=1).reset_index().rename(columns={"index": "team"})
    return ratings

team_epa = build_team_epa(pbp)

# --- Player Usage ---
@st.cache_data
def build_usage(player_stats):
    df = player_stats.copy()
    df = df[df["season_type"] == "REG"]
    # Target share
    team_targets = df.groupby(["team", "week"])["targets"].sum().reset_index(name="team_targets")
    df = df.merge(team_targets, on=["team", "week"], how="left")
    df["target_share"] = df["targets"] / df["team_targets"].replace(0, np.nan)
    # Rush share
    team_carries = df.groupby(["team", "week"])["carries"].sum().reset_index(name="team_carries")
    df = df.merge(team_carries, on=["team", "week"], how="left")
    df["rush_share"] = df["carries"] / df["team_carries"].replace(0, np.nan)
    # Recent 4 weeks average
    recent = (
        df.sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(4)
        .groupby("player_id")
        .agg(
            player_name=("player_display_name", "last"),
            team=("team", "last"),
            position=("position", "last"),
            target_share=("target_share", "mean"),
            rush_share=("rush_share", "mean"),
            targets=("targets", "mean"),
            carries=("carries", "mean"),
        )
        .reset_index()
    )
    return recent

usage = build_usage(player_stats)

# --- UI ---
teams = sorted(team_epa["team"].unique())
col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Home Team", teams, index=teams.index("BAL") if "BAL" in teams else 0)
with col2:
    away = st.selectbox("Away Team", teams, index=teams.index("JAX") if "JAX" in teams else 1)

n_sims = st.sidebar.slider("Simulations", 2000, 15000, 8000, 1000)
total_line = st.sidebar.number_input("Total Line", value=48.5, step=0.5)
spread = st.sidebar.number_input("Home Spread (negative = home favorite)", value=-3.5, step=0.5)

if st.button("Run Simulation", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    # Get ratings
    home_off = team_epa.loc[team_epa["team"] == home, "off_epa"].values[0]
    home_def = team_epa.loc[team_epa["team"] == home, "def_epa"].values[0]
    away_off = team_epa.loc[team_epa["team"] == away, "off_epa"].values[0]
    away_def = team_epa.loc[team_epa["team"] == away, "def_epa"].values[0]

    # Expected points adjustment (simple EPA model)
    home_exp = 22.5 + (home_off - away_def) * 12
    away_exp = 21.0 + (away_off - home_def) * 12

    home_scores = []
    away_scores = []
    totals = []

    for i in range(n_sims):
        if i % 500 == 0:
            progress.progress(i / n_sims)
            status.text(f"Simulating... {i}/{n_sims}")
        hs = max(0, np.random.normal(home_exp, 9.5))
        as_ = max(0, np.random.normal(away_exp, 9.5))
        home_scores.append(hs)
        away_scores.append(as_)
        totals.append(hs + as_)

    progress.progress(1.0)
    status.empty()

    home_scores = np.array(home_scores)
    away_scores = np.array(away_scores)
    totals = np.array(totals)

    # Results
    st.subheader("Simulation Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Home Win %", f"{(home_scores > away_scores).mean():.1%}")
    c2.metric("Away Win %", f"{(away_scores > home_scores).mean():.1%}")
    c3.metric("Over %", f"{(totals > total_line).mean():.1%}")
    c4.metric("Home Cover %", f"{(home_scores - away_scores > spread).mean():.1%}")

    st.write(f"**Expected Score:** {home} {home_scores.mean():.1f} - {away_scores.mean():.1f} {away}")
    st.write(f"**Expected Total:** {totals.mean():.1f}")

    # Distribution chart
    fig = px.histogram(totals, nbins=40, title="Total Points Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Player Props (usage-based)
    st.subheader("Player Props (Usage-Adjusted)")
    for team in [home, away]:
        st.write(f"**{team}**")
        team_players = usage[usage["team"] == team].sort_values("target_share", ascending=False).head(6)
        props = []
        for _, p in team_players.iterrows():
            td_prob = min(0.55, 0.12 + (p["target_share"] or 0) * 1.8 + (p["rush_share"] or 0) * 1.4)
            rec_yards = (p["targets"] or 0) * 7.5 * (1 + (p["target_share"] or 0))
            rush_yards = (p["carries"] or 0) * 4.2 * (1 + (p["rush_share"] or 0))
            props.append({
                "Player": p["player_name"],
                "Pos": p["position"],
                "Anytime TD %": f"{td_prob:.1%}",
                "Est. Rec Yds": f"{rec_yards:.0f}",
                "Est. Rush Yds": f"{rush_yards:.0f}",
                "Tgt Share": f"{(p['target_share'] or 0):.1%}",
                "Rush Share": f"{(p['rush_share'] or 0):.1%}",
            })
        st.dataframe(pd.DataFrame(props), use_container_width=True, hide_index=True)

    # Live single-game simulation
    if st.button("Watch One Simulated Game"):
        st.subheader("Live Simulated Game")
        log = st.empty()
        h_score, a_score = 0, 0
        plays = []
        for q in range(1, 5):
            for drive in range(3):
                if np.random.rand() < 0.5:
                    pts = np.random.choice([0, 3, 7], p=[0.35, 0.25, 0.40])
                    h_score += pts
                    plays.append(f"Q{q}: {home} scores {pts} pts → {h_score}-{a_score}")
                else:
                    pts = np.random.choice([0, 3, 7], p=[0.40, 0.25, 0.35])
                    a_score += pts
                    plays.append(f"Q{q}: {away} scores {pts} pts → {h_score}-{a_score}")
                log.text("\n".join(plays[-8:]))
                st.rerun()  # forces visual update in some Streamlit versions
        st.success(f"Final: {home} {h_score} - {a_score} {away}")