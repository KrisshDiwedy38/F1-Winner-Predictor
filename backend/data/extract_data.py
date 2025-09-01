import fastf1 as f1
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase : Client = create_client(supabase_url,supabase_key)

f1.Cache.enable_cache('backend/data/f1cache')


years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
events = ['R']
rounds = [21, 21, 17, 22, 22 ,22 ,24]

result_values = []
weather_values = []

count = -1
for year in years:
   count += 1
   for race_round in range(1,rounds[count] + 1):
      session = f1.get_session(year, race_round, 'R')
      session.load()
      # Set a unique id for Primary Key and referencing other tables
      race_id = f"{str(year)}_{race_round}"
      # Extracting race name
      race_name = str(session).split(":")[1].split("-")[0].strip()
      # Checking if the race had rain
      rainfall = 1 if session.weather_data["Rainfall"].any() else 0
      
      # Getting required data
      result_data = session.results.loc[:,['TeamName','Abbreviation','FullName','Time', 'Position', 'Status']].copy()
      # Process each driver's result
      for _, row in result_data.iterrows():
        # Handling null time values given to lapped cars 
        if row.Status == "+1 Lap":
            time += 100.00
        elif row.Status == "+2 Laps":
            time += 200.00
        else: 
            # Converting timedelta value to float value(in seconds)
            time += (row.Time).total_seconds()
        
        race_data = {
            'position': int(row['Position']) if not pd.isnull(row['Position']) else None,
            'raceid': race_id,
            'racename': race_name,
            'teamname': row['TeamName'],
            'drivercode': row['Abbreviation'],
            'fullname': row['FullName'],
            'timesecs': round(time,4),
            'status': row['Status']
        }
        result_values.append(race_data)
      
      # Store weather data (one entry per race)
      weather_data = {
            'raceid': race_id,
            'racename': race_name,
            'rainfall': rainfall
      }
      weather_values.append(weather_data)
      
      print(f"{race_id} - {race_name} completed")

# Converting values into a panads dataframe
race_df = pd.DataFrame(result_values)
weather_df = pd.DataFrame(weather_values)

race_df = race_df.replace({np.nan: None})
weather_df = weather_df.replace({np.nan: None})

# Fix missing positions
race_df['position'] = race_df['position'].fillna(-1).astype(int)

race_dict = race_df.to_dict(orient="records")
weather_dict = weather_df.to_dict(orient="records")

batch_size = 500
for i in range(0, len(race_dict), batch_size):
    supabase.table('races').upsert(race_dict[i:i+batch_size]).execute()

print("\nRace Data Uploaded")

for i in range(0, len(weather_dict), batch_size):
    supabase.table('weather').upsert(weather_dict[i:i+batch_size]).execute()

print("\nWeather Data Uploaded")
