import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

class AccurateMultiSportSimulator:
    def __init__(self):
        # NFL Power Ratings (approximate)
        self.nfl_teams = {
            'Cardinals': 0.5, 'Falcons': 1.2, 'Ravens': 3.8, 'Bills': 3.5,
            'Panthers': -1.5, 'Bears': 1.8, 'Bengals': 2.4, 'Browns': 0.8,
            'Cowboys': 1.5, 'Broncos': 2.0, 'Lions': 3.2, 'Packers': 2.8,
            'Texans': 2.6, 'Colts': 1.0, 'Jaguars': 0.4, 'Chiefs': 4.2,
            'Raiders': -0.8, 'Chargers': 1.8, 'Rams': 2.2, 'Dolphins': 0.6,
            'Vikings': 2.5, 'Patriots': 0.3, 'Saints': 1.4, 'Giants': -0.5,
            'Jets': 0.2, 'Eagles': 3.6, 'Steelers': 1.6, '49ers': 3.0,
            'Seahawks': 1.9, 'Buccaneers': 1.3, 'Titans': -0.4, 'Commanders': 2.1
        }

        # NBA Offensive Ratings / PPG (approximate)
        self.nba_teams = {
            'Hawks': 116.5, 'Celtics': 118.8, 'Nets': 114.2, 'Hornets': 112.8,
            'Bulls': 113.5, 'Cavaliers': 117.2, 'Mavericks': 116.8, 'Nuggets': 117.5,
            'Pistons': 111.5, 'Warriors': 115.0, 'Rockets': 114.8, 'Pacers': 116.0,
            'Clippers': 115.5, 'Lakers': 115.8, 'Grizzlies': 113.8, 'Heat': 112.5,
            'Bucks': 116.2, 'Timberwolves': 115.2, 'Pelicans': 114.0, 'Knicks': 115.5,
            'Thunder': 117.8, 'Magic': 113.2, '76ers': 114.5, 'Suns': 115.0,
            'Trail Blazers': 112.0, 'Kings': 116.5, 'Spurs': 114.2, 'Raptors': 113.0,
            'Jazz': 112.5, 'Wizards': 111.8
        }

        # MLB Runs per Game (approximate)
        self.mlb_teams = {
            'Diamondbacks': 4.6, 'Braves': 4.9, 'Orioles': 4.5, 'Red Sox': 4.7,
            'Cubs': 4.6, 'White Sox': 3.8, 'Reds': 4.5, 'Guardians': 4.4,
            'Rockies': 4.2, 'Tigers': 4.5, 'Astros': 4.7, 'Royals': 4.3,
            'Angels': 4.1, 'Dodgers': 5.1, 'Marlins': 4.0, 'Brewers': 4.6,
            'Twins': 4.5, 'Mets': 4.6, 'Yankees': 4.8, 'Athletics': 4.2,
            'Phillies': 4.7, 'Pirates': 4.1, 'Padres': 4.5, 'Giants': 4.3,
            'Mariners': 4.4, 'Cardinals': 4.4, 'Rays': 4.3, 'Rangers': 4.6,
            'Blue Jays': 4.5, 'Nationals': 4.2
        }

    def _adjust_nfl_margin_for_key_numbers(self, raw_margin):
        if abs(raw_margin - 3) < 1.2 and np.random.rand() < 0.35:
            return 3 if raw_margin > 0 else -3
        if abs(raw_margin - 7) < 1.5 and np.random.rand() < 0.28:
            return 7 if raw_margin > 0 else -7
        if abs(raw_margin - 10) < 1.0 and np.random.rand() < 0.12:
            return 10 if raw_margin > 0 else -10
        return round(raw_margin)

    def simulate_nfl(self, home, away, n_sims=30000, home_adv=2.1, sigma=13.6):
        home_pwr = self.nfl_teams.get(home, 0.0)
        away_pwr = self.nfl_teams.get(away, 0.0)
        home_scores, away_scores = [], []
        
        for _ in range(n_sims):
            exp_margin = (home_pwr + home_adv) - away_pwr + np.random.normal(0, 0.8)
            raw_margin = np.random.normal(exp_margin, sigma)
            margin = self._adjust_nfl_margin_for_key_numbers(raw_margin)
            
            total = np.random.normal(46.8 + 0.2*(home_pwr + away_pwr), 9.0)
            total = max(28, total)
            
            hs = max(0, round((total + margin) / 2))
            aws = max(0, round((total - margin) / 2))
            
            if abs((hs - aws) - margin) > 1:
                diff = margin - (hs - aws)
                hs = max(0, hs + diff // 2)
                aws = max(0, aws - diff // 2)
            
            home_scores.append(hs)
            away_scores.append(aws)
        
        return pd.DataFrame({
            'home_score': home_scores,
            'away_score': away_scores,
            'margin': np.array(home_scores) - np.array(away_scores),
            'total': np.array(home_scores) + np.array(away_scores),
            'home_win': np.array(home_scores) > np.array(away_scores)
        })

    def simulate_nba(self, home, away, n_sims=25000, home_adv=2.0):
        h_off = self.nba_teams.get(home, 114.5)
        a_off = self.nba_teams.get(away, 114.5)
        home_scores, away_scores = [], []
        
        for _ in range(n_sims):
            shared_noise = np.random.normal(0, 4)
            h_exp = h_off + home_adv + shared_noise + np.random.normal(0, 7)
            a_exp = a_off + shared_noise + np.random.normal(0, 7)
            
            hs = int(np.clip(np.random.normal(h_exp, 9.5), 85, 145))
            aws = int(np.clip(np.random.normal(a_exp, 9.5), 85, 145))
            home_scores.append(hs)
            away_scores.append(aws)
        
        return pd.DataFrame({
            'home_score': home_scores,
            'away_score': away_scores,
            'margin': np.array(home_scores) - np.array(away_scores),
            'total': np.array(home_scores) + np.array(away_scores),
            'home_win': np.array(home_scores) > np.array(away_scores)
        })

    def simulate_mlb(self, home, away, n_sims=25000, home_adv=0.22):
        h_runs = self.mlb_teams.get(home, 4.5)
        a_runs = self.mlb_teams.get(away, 4.5)
        home_scores, away_scores = [], []
        
        for _ in range(n_sims):
            hs = poisson.rvs(max(1.8, h_runs + home_adv + np.random.normal(0, 0.3)))
            aws = poisson.rvs(max(1.8, a_runs + np.random.normal(0, 0.3)))
            home_scores.append(hs)
            away_scores.append(aws)
        
        return pd.DataFrame({
            'home_score': home_scores,
            'away_score': away_scores,
            'margin': np.array(home_scores) - np.array(away_scores),
            'total': np.array(home_scores) + np.array(away_scores),
            'home_win': np.array(home_scores) > np.array(away_scores)
        })

st.set_page_config(page_title="Multi-Sport Simulator", layout="centered")
st.title("Multi-Sport Monte Carlo Simulator")

sim = AccurateMultiSportSimulator()

sport = st.selectbox("Choose Sport", ["NFL", "NBA", "MLB"])
n_sims = st.slider("Number of Simulations", min_value=10000, max_value=100000, value=50000, step=10000)

if sport == "NFL":
    teams = sorted(list(sim.nfl_teams.keys()))
elif sport == "NBA":
    teams = sorted(list(sim.nba_teams.keys()))
else:
    teams = sorted(list(sim.mlb_teams.keys()))

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
            st.write(f"**|7| Margin Frequency:** {(df['margin'].abs() == 7).mean():.1%}")