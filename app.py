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

        # NFL Player Props (2026-27 key players)
        self.nfl_players = {
            'Bears': {
                'Caleb Williams': {'pass_yds': 250, 'pass_tds': 1.7},
                'D\'Andre Swift': {'rush_yds': 75, 'rush_tds': 0.5},
                'Rome Odunze': {'rec_yds': 65, 'receptions': 4.8}
            },
            'Patriots': {
                'Drake Maye': {'pass_yds': 240, 'pass_tds': 1.6},
                'Rhamondre Stevenson': {'rush_yds': 70, 'rush_tds': 0.5},
                'A.J. Brown': {'rec_yds': 75, 'receptions': 5.2}
            },
            'Chiefs': {
                'Patrick Mahomes': {'pass_yds': 275, 'pass_tds': 2.0},
                'Kenneth Walker III': {'rush_yds': 85, 'rush_tds': 0.65},
                'Travis Kelce': {'rec_yds': 60, 'receptions': 5.0}
            },
            'Eagles': {
                'Saquon Barkley': {'rush_yds': 90, 'rush_tds': 0.8},
                'DeVonta Smith': {'rec_yds': 70, 'receptions': 5.0}
            },
            'Bills': {
                'Josh Allen': {'pass_yds': 270, 'pass_tds': 2.0},
                'James Cook': {'rush_yds': 80, 'rush_tds': 0.6}
            },
            'Ravens': {
                'Lamar Jackson': {'pass_yds': 230, 'pass_tds': 1.7},
                'Derrick Henry': {'rush_yds': 85, 'rush_tds': 0.75}
            },
            'Lions': {
                'Jared Goff': {'pass_yds': 255, 'pass_tds': 1.8},
                'Jahmyr Gibbs': {'rush_yds': 80, 'rush_tds': 0.7}
            },
            'Packers': {
                'Jordan Love': {'pass_yds': 260, 'pass_tds': 1.8},
                'Josh Jacobs': {'rush_yds': 75, 'rush_tds': 0.6}
            },
            'Cowboys': {
                'Dak Prescott': {'pass_yds': 260, 'pass_tds': 1.8},
                'CeeDee Lamb': {'rec_yds': 85, 'receptions': 6.5}
            },
            '49ers': {
                'Brock Purdy': {'pass_yds': 245, 'pass_tds': 1.6},
                'Christian McCaffrey': {'rush_yds': 85, 'rush_tds': 0.8}
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
                'Jayson Tatum': {'points': 26.5, 'rebounds': 8.0, 'assists': 4.5},
                'Derrick White': {'points': 16.0, 'rebounds': 4.0, 'assists': 4.5}
            },
            'Lakers': {
                'Luka Dončić': {'points': 28.5, 'rebounds': 8.5, 'assists': 8.0},
                'Austin Reaves': {'points': 18.0, 'rebounds': 4.5, 'assists':