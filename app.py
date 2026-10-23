import streamlit as st
from simulator import AccurateMultiSportSimulator

st.set_page_config(page_title="Multi-Sport Simulator", layout="centered")
st.title("Multi-Sport Monte Carlo Simulator")

sim = AccurateMultiSportSimulator()

sport = st.selectbox("Choose Sport", ["NFL", "NBA", "MLB"])
n_sims = st.slider("Number of Simulations", min_value=10000, max_value=100000, value=50000, step=10000)

if sport == "NFL":
    teams = list(sim.nfl_teams.keys())
elif sport == "NBA":
    teams = list(sim.nba_teams.keys())
else:
    teams = list(sim.mlb_teams.keys())

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Home Team", teams)
with col2:
    away = st.selectbox("Away Team", teams)

if st.button("Run Simulation", type="primary"):
    with st.spinner("Running simulations..."):
        if sport == "NFL":
            df = sim.simulate_nfl(home, away, n_sims=n_sims)
        elif sport == "NBA":
            df = sim.simulate_nba(home, away, n_sims=n_sims)
        else:
            df = sim.simulate_mlb(home, away, n_sims=n_sims)

        win_pct = df["home_win"].mean()
        st.success(f"**{home} Win Probability: {win_pct:.1%}**")
        st.write(f"**Average Score:** {df['home_score'].mean():.1f} – {df['away_score'].mean():.1f}")
        st.write(f"**Median Score:** {df['home_score'].median():.0f} – {df['away_score'].median():.0f}")

        if sport == "NFL":
            st.write(f"**|3| Margin Frequency:** {(df['margin'].abs() == 3).mean():.1%}")
            st.write(f"**|7| Margin