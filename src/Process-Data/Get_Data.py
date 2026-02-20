import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame

config = toml.load("config.toml")
url = config['data_url']
con = sqlite3.connect("../../Data/TeamData.sqlite")

def table_exists(table_name, connection):
    """Checks if a specific table exists in the SQLite database."""
    cursor = connection.cursor()
    # Query the system master table for the table name
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    cursor.execute(query, (table_name,))
    return cursor.fetchone() is not None

for key, value in config['get-data'].items():
    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()

    while date_pointer <= end_date:
        # Define the table name (the date)
        table_name = date_pointer.strftime("%Y-%m-%d")

        # --- NEW: Check if we already have this data ---
        if table_exists(table_name, con):
            print(f"Skipping {table_name}: Data already exists.")
            date_pointer = date_pointer + timedelta(days=1)
            continue 
        # -----------------------------------------------

        print("Getting data: ", date_pointer)

        try:
            raw_data = get_json_data(
                url.format(date_pointer.month, date_pointer.day, value['start_year'], date_pointer.year, key))
            df = to_data_frame(raw_data)

            # Note: You were incrementing date_pointer BEFORE saving it as a column
            # Moving the increment after the df processing for logic clarity
            df['Date'] = str(date_pointer)
            df.to_sql(table_name, con, if_exists="replace")

            time.sleep(random.randint(1, 3))
        except Exception as e:
            print(f"Error fetching data for {date_pointer}: {e}")

        date_pointer = date_pointer + timedelta(days=1)

con.close()