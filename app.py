import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

class AccurateMultiSportSimulator:
    def __init__(self):
        # NFL Power Ratings
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

        # NFL Player Props (key players)
        self.nfl_players = {
            'Bears': {
                'Caleb Williams': {'pass_yds': 245, 'pass_tds': 1.6},
                'D\'Andre Swift': {'rush_yds': 78, 'rush_tds': 0.55},
                'DJ Moore': {'rec_yds': 68, 'receptions': 5.2}
            },
            'Patriots': {
                'Drake Maye': {'pass_yds': 230, 'pass_tds': 1.5},
                'Rhamondre Stevenson': {'rush_yds': 72, 'rush_tds': 0.5},
                'Hunter Henry': {'rec_yds': 55, 'receptions': 4.8}
            },
            'Chiefs': {
                'Patrick Mahomes': {'pass_yds': 280, 'pass_tds': 2.1},
                'Isiah Pacheco': {'rush_yds': 75, 'rush_tds': 0.6},
                'Travis Kelce': {'rec_yds': 70, 'receptions': 5.5}
            },
            'Eagles': {
                'Jalen Hurts': {'pass_yds': 250, 'pass_tds': 1.8},
                'Saquon Barkley': {'rush_yds': 95, 'rush_tds': 0.85},
                'A.J. Brown': {'rec_yds': 75, 'receptions': 5.0}
            },
            'Bills': {
                'Josh Allen': {'pass_yds': 275, 'pass_tds': 2.0},
                'James Cook': {'rush_yds': 80, 'rush_tds': 0.6},
                'Stefon Diggs': {'rec_yds': 72, 'receptions': 5.3}
            },
            'Ravens': {
                'Lamar Jackson': {'pass_yds': 230, 'pass_tds': 1.7},
                'Derrick Henry': {'rush_yds': 90, 'rush_tds': 0.8},
                'Zay Flowers': {'rec_yds': 65, 'receptions': 4.9}
            }
        }

        # NBA Teams
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

        # NBA Player Props
        self.nba_players = {
            'Celtics': {
                'Jayson Tatum': {'points': 27.5, 'rebounds': 8.5, 'assists': 4.8},
                'Jaylen Brown': {'points': 23.5, 'rebounds': 5.5, 'assists': 3.5}
            },
            'Lakers': {
                'LeBron James': {'points': 24.5, 'rebounds': 7.5, 'assists': 8.2},
                'Anthony Davis': {'points': 25.0, 'rebounds': 11.5, 'assists': 3.2}
            },
            'Nuggets': {
                'Nikola Jokic': {'points': 26.5, 'rebounds': 12.5, 'assists': 9.0},
                'Jamal Murray': {'points': 21.5, 'rebounds': 4.0, 'assists': 6.5}
            },
            'Warriors': {
                'Stephen Curry': {'points': 26.0, 'rebounds': 4.5, 'assists': 5.0},
                'Draymond Green': {'points': 9.5, 'rebounds': 7.0, 'assists': 6.0}
            },
            'Bucks': {
                'Giannis Antetokounmpo': {'points': 30.5, 'rebounds': 11.5, 'assists': 6.5}
            },
            'Thunder': {
                'Shai Gilgeous-Alexander': {'points': 30.0, 'rebounds': 5.5, 'assists': 6.2}
            }
        }

        # MLB Teams
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

        # MLB Player Props
        self.mlb_players = {
            'Yankees': {
                'Aaron Judge': {'hits': 1.1, 'home_runs': 0.55, 'rbis': 1.2},
                'Juan Soto': {'hits': 1.05, 'home_runs': 0.45, 'rbis': 1.0}
            },
            'Dodgers': {
                'Shohei Ohtani': {'hits': 1.15, 'home_runs': 0.6, 'rbis': 1.25},
                'Mookie Betts': {'hits': 1.1, 'home_runs': 0.4, 'rbis': 0.9}
            },
            'Braves': {
                'Ronald Acuña Jr.': {'hits': 1.2, 'home_runs': 0.5, 'rbis': 1.1}
            },
            'Astros': {
                'Jose Altuve': {'hits': 1.15, 'home_runs': 0.3, 'rbis': 0.8}
            },
            'Phillies': {
                'Bryce Harper': {'hits': 1.1, 'home_runs': 0.5, 'rbis': 1.15}
            },
            'Padres': {
                'Fernando Tatis Jr.': {'hits': 1.05, 'home_runs': 0.45, 'rbis': 1.0}
            }
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
            'home_score': home_scores, 'away_score': away_scores,
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
            'home_score': home_scores, 'away_score': away_scores,
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
            'home_score': home_scores, 'away_score': away_scores,
            'margin': np.array(home_scores) - np.array(away_scores),
            'total': np.array(home_scores) + np.array(away_scores),
            'home_win': np.array(home_scores) > np.array(away_scores)
        })

    def simulate_player_props(self, sport, team, n_sims):
        if sport == "NFL":
            players = self.nfl_players.get(team, {})
        elif sport == "NBA":
            players = self.nba_players.get(team, {})
        else:
            players = self.mlb_players.get(team, {})
        
        if not players:
            return None
        
        results = {}
        for player, stats in players.items():
            player_results = {}
            for stat, mean in stats.items():
                if any(x in stat for x in ['yds', 'points', 'rebounds', 'assists']):
                    # Continuous stats
                    std = mean * 0.25 if sport == "NFL" else mean * 0.3
                    vals = np.clip(np.random.normal(mean, std, n_sims), 0, None)
                else:
                    # Counting stats (TDs, hits, HRs, etc.)
                    vals = poisson.rvs(mean, size=n_sims)
                player_results[stat] = vals
            results[player] = player_results
        return results

# Streamlit UI
st.set_page_config(page_title="Multi-Sport Simulator", layout="centered")
st.title("Multi-Sport Simulator + Player Props")

sim = AccurateMultiSportSimulator()

sport = st.selectbox("Sport", ["NFL", "NBA", "MLB"])
n_sims = st.slider("Simulations", 10000, 100000, 50000, 10000)

if sport == "NFL":
    teams = sorted(sim.nfl_teams.keys())
elif sport == "NBA":
    teams = sorted(sim.nba_teams.keys())
else:
    teams = sorted(sim.mlb_teams.keys())

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Home", teams)
with col2:
    away = st.selectbox("Away", teams)

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        if sport == "NFL":
            df = sim.simulate_nfl(home, away, n_sims)
        elif sport == "NBA":
            df = sim.simulate_nba(home, away, n_sims)
        else:
            df = sim.simulate_mlb(home, away, n_sims)

        home_props = sim.simulate_player_props(sport, home, n_sims)
        away_props = sim.simulate_player_props(sport, away, n_sims)

        st.success(f"**{home} Win Probability: {df['home_win'].mean():.1%}**")
        st.write(f"**Average Score:** {df['home_score'].mean():.1f} – {df['away_score'].mean():.1f}")

        if sport == "NFL":
            st.write(f"**|3| Frequency:** {(df['margin'].abs() == 3).mean():.1%}")
            st.write(f"**|7| Frequency:** {(df['margin'].abs() == 7).mean():.1%}")

        st.subheader("Player Props (Simulated Averages)")
        for team_name, props in [(home, home_props), (away, away_props)]:
            if props:
                st.write(f"**{team_name}**")
                for player, stats in props.items():
                    for stat, values in stats.items():
                        st.write(f"- {player} {stat.replace('_', ' ').title()}: {values.mean():.1f}")