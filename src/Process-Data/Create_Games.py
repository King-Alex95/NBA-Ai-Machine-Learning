import os
import sqlite3
import sys
import numpy as np
import pandas as pd
import toml

# Add path for custom utility imports
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.Dictionaries import (team_index_07, team_index_08, team_index_12, 
                                     team_index_13, team_index_14, team_index_current)

config = toml.load("config.toml")

# 1. SETUP DATABASE CONNECTIONS
hist_db_path = os.path.abspath("../../Data/OddsData.sqlite")
recent_db_path = os.path.abspath("/Users/kingalex/Desktop/NBA-Machine-Learning-Sports-Betting/src/Process-Data/Data/OddsData.sqlite")
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")

# Connect to Historical and ATTACH Recent
odds_con = sqlite3.connect(hist_db_path)
odds_con.execute(f"ATTACH DATABASE '{recent_db_path}' AS recent_db")

# Data Staging Lists
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []

for key in config['create-games'].keys():
    print(f"Processing Season: {key}")
    odds_df = None

    # SMART SEARCH: Check historical naming, then recent naming
    queries = [
        f"SELECT * FROM main.\"odds_{key}_new\"", 
        f"SELECT * FROM recent_db.\"{key}\"",     
        f"SELECT * FROM main.\"{key}\""           
    ]

    for q in queries:
        try:
            # We don't use index_col="index" here because new tables might not have that index
            odds_df = pd.read_sql_query(q, odds_con)
            break 
        except:
            continue

    if odds_df is None:
        print(f"!!! Could not find table for {key}. Skipping.")
        continue

    # PROCESS GAMES IN SEASON
    for row in odds_df.itertuples():
        # Adjusting tuple indices based on your table structure
        # (Assuming: 1:Date, 2:Home, 3:Away, 4:OU, 8:Points, 9:WinMargin, 10:RestHome, 11:RestAway)
        try:
            date = row.Date
            home_team = row.Home
            away_team = row.Away

            team_df = pd.read_sql_query(f"select * from \"{date}\"", teams_con)
            
            if len(team_df.index) == 30:
                scores.append(row.Points)
                OU.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                if row.Points < row.OU:
                    OU_Cover.append(0)
                elif row.Points > row.OU:
                    OU_Cover.append(1)
                else:
                    OU_Cover.append(2)

                # ERA INDEXING LOGIC
                season = key
                if season == '2007-08':
                    idx = team_index_07
                elif season in ["2008-09", "2009-10", "2010-11", "2011-12"]:
                    idx = team_index_08
                elif season == "2012-13":
                    idx = team_index_12
                elif season == '2013-14':
                    idx = team_index_13
                elif season in ['2022-23', '2023-24', '2024-25', '2025-26']:
                    idx = team_index_current
                else:
                    idx = team_index_14

                home_team_series = team_df.iloc[idx.get(home_team)]
                away_team_series = team_df.iloc[idx.get(away_team)]
                
                # Merge Home and Away stats into one row
                game = pd.concat([home_team_series, away_team_series.rename(
                    index={col: f"{col}.1" for col in team_df.columns.values}
                )])
                games.append(game)
        except Exception as e:
            # Skip games where stats are missing
            continue

# BUILD FINAL DATASET
if not games:
    print("No data processed. Check database connections.")
    sys.exit()

season_data = pd.concat(games, ignore_index=True, axis=1).T
frame = season_data.drop(columns=['TEAM_ID', 'TEAM_ID.1'], errors='ignore')

# Add targets
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)

# Final clean: convert all numeric columns to float
for field in frame.columns.values:
    if 'TEAM_' in field or 'Date' in field:
        continue
    frame[field] = pd.to_numeric(frame[field], errors='coerce').astype(float)

# SAVE TO DISK
dataset_con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2007_2026_new", dataset_con, if_exists="replace")

odds_con.close()
teams_con.close()
dataset_con.close()
print("Process Complete. Dataset saved as 'dataset_2007_2026_new' in dataset.sqlite")