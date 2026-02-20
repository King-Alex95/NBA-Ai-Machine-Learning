import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import toml
from sbrscrape import Scoreboard

# Ensure we are in the right directory
db_dir = "Data"
db_name = "OddsData.sqlite"
db_path = os.path.join(db_dir, db_name)
os.makedirs(db_dir, exist_ok=True)

# 1. FIX: Use isolation_level=None to ensure data writes immediately
con = sqlite3.connect(db_path, isolation_level=None)
print(f"Database connected at: {os.path.abspath(db_path)}")

sportsbook = 'fanduel'
config = toml.load("config.toml")

for key, value in config['get-odds-data'].items():
    # --- RESUME LOGIC ---
    try:
        # Check if table exists and get last date
        last_date_df = pd.read_sql(f"SELECT MAX(Date) as last_date FROM '{key}'", con)
        last_date_str = last_date_df.iloc[0]['last_date']
        
        if last_date_str:
            # Handle potential timestamp strings from SQLite
            date_part = last_date_str.split()[0] 
            start_date = datetime.strptime(date_part, "%Y-%m-%d").date() + timedelta(days=1)
            print(f"Resuming {key} from: {start_date}")
        else:
            start_date = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    except Exception:
        print(f"No existing table for {key}. Starting fresh.")
        start_date = datetime.strptime(value['start_date'], "%Y-%m-%d").date()

    date_pointer = start_date
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()

    while date_pointer <= end_date:
        print(f"Checking: {date_pointer}...", end=" ", flush=True)
        df_day_data = []
        
        try:
            sb = Scoreboard(date=date_pointer)
            if hasattr(sb, "games") and sb.games:
                for game in sb.games:
                    try:
                        df_day_data.append({
                            'Date': date_pointer.isoformat(), # Save as string for SQLite compatibility
                            'Home': game['home_team'],
                            'Away': game['away_team'],
                            'OU': game.get('total', {}).get(sportsbook),
                            'Spread': game.get('away_spread', {}).get(sportsbook),
                            'ML_Home': game.get('home_ml', {}).get(sportsbook),
                            'ML_Away': game.get('away_ml', {}).get(sportsbook),
                            'Points': (game.get('away_score', 0) or 0) + (game.get('home_score', 0) or 0),
                            'Win_Margin': (game.get('home_score', 0) or 0) - (game.get('away_score', 0) or 0),
                            'Days_Rest_Home': 7, 
                            'Days_Rest_Away': 7
                        })
                    except Exception as game_err:
                        continue # Skip individual broken games
        except Exception as e:
            print(f"Network error: {e}")
            time.sleep(5)
            continue

        # --- SAVE IMMEDIATELY ---
        if df_day_data:
            df_day = pd.DataFrame(df_day_data)
            # 2. FIX: Use 'append' and explicit commit
            df_day.to_sql(key, con, if_exists="append", index=False)
            con.commit() 
            print(f"Saved {len(df_day_data)} games.")
        else:
            print("No games found.")

        date_pointer += timedelta(days=1)
        time.sleep(random.uniform(1.5, 3.5))

con.close()
print("Process finished.")