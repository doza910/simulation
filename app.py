import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

class AccurateMultiSportSimulator:
    def __init__(self):
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

        self.nfl_players = {
            'Bears': {
                'Caleb Williams': {'pass_yds': 250, 'pass_tds': 1.7},
                'D\'Andre Swift': {'rush_yds': 75, 'rush_tds': 0.5},
                'Rome Odunze': {'rec_yds': 65, 'receptions': 4.8}
            },
            'Patriots': {
                'Drake Maye': {'pass_yds': 240, 'pass_tds': 1.6},
                'Rhamondre Stevenson': {'rush_yds': 70, 'rush_tds': 0.5}
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
                'CeeDee Lamb': {'rec_yds': 85, 'receptions':