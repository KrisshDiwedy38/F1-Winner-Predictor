import fastf1 as f1
import pandas as pd
import numpy as np
from datetime import date
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from get_supabase_data import load_data

h_race_df, h_weather_df = load_data()

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase : Client = create_client(supabase_url, supabase_key)

f1.Cache.enable_cache('backend/data/f1cache')

def completed_rounds(year):
   """Get number of completed races for a given year"""
   schedule = f1.get_event_schedule(year)
   completed = -1
   
   for race_date in schedule['EventDate']:
      datestamp = pd.Timestamp(race_date)
      if datestamp.date() < date.today():
         completed += 1
   
   return completed

def get_recent_data(current_year):
   """Extract recent F1 race data and weather information"""
   
   result_values = []
   weather_values = []

   completed_races = completed_rounds(current_year)
   
   if completed_races == 0:
      print(f"No completed races found for {current_year}")
      return current_year

   # Get last race info from database
   last_race_id = h_race_df['raceid'].iloc[-1] if not h_race_df.empty else "0_0"
   
   last_entry_year = int(last_race_id.split("_")[0])
   last_entry_race = int(last_race_id.split("_")[1])
   
   # Determine starting point for data extraction
   if last_entry_year == current_year and last_entry_race >= completed_races:    
      print("Data Already Up to Date")
      return current_year
   
   # Calculate starting race number
   if last_entry_year != current_year:
      last_season_races = completed_rounds(last_entry_year)
      print(last_season_races)
      print(last_entry_race)
      if last_entry_race == last_season_races:
         start_race = 1
      else:
         current_year = last_entry_year
         start_race = last_entry_race + 1
   else:
      start_race = last_entry_race + 1

   print(f"Fetching data from {current_year} round {start_race} to {completed_races}")

   try:
      # Extract data for each missing race
      for race_round in range(start_race, completed_races + 1):
         print(f"Processing {current_year} Round {race_round}...")
         
         session = f1.get_session(current_year, race_round, 'R')
         session.load()

         # Extract race metadata
         race_id = f"{current_year}_{race_round}"
         race_name = str(session).split(":")[1].split("-")[0].strip()

         # Check for rainfall
         rainfall = 1 if session.weather_data["Rainfall"].any() else 0
         
         # Get race results
         result_data = session.results.loc[:,['TeamName','Abbreviation','FullName','Time', 'Position', 'Status']].copy()
         time = 0
         
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

   except Exception as e:
      print(f"Error occurred while fetching data: {e}")
      print(f"Failed at {current_year} Round {race_round}")
      return current_year

   # Convert to DataFrames
   race_df = pd.DataFrame(result_values)
   weather_df = pd.DataFrame(weather_values)

   print(f"\nExtracted {len(race_df)} race results and {len(weather_df)} weather records")
   print("\nSample Race Data:")
   print(race_df)
   print("\nWeather Data:")
   print(weather_df)

   # Upload to Supabase
   if not race_df.empty:
      try:
         race_df = race_df.replace({np.nan: None})
         weather_df = weather_df.replace({np.nan: None}) 

         # Fix missing positions
         race_df['position'] = race_df['position'].fillna(-1).astype(int) 

         race_dict = race_df.to_dict(orient="records")
         weather_dict = weather_df.to_dict(orient="records")  

         batch_size = 500
         
         # Upload race data in batches
         for i in range(0, len(race_dict), batch_size):
               batch = race_dict[i:i+batch_size]
               supabase.table('races').upsert(batch).execute()
         print("✓ Race data uploaded to Supabase")

         # Upload weather data in batches
         for i in range(0, len(weather_dict), batch_size):
               batch = weather_dict[i:i+batch_size]
               supabase.table('weather').upsert(batch).execute()
         print("✓ Weather data uploaded to Supabase")
         
      except Exception as e:
         print(f"Error uploading to Supabase: {e}")
         return False

   return current_year

def main():
   current_year = date.today().year
   
   # Get recent data for current year
   year = get_recent_data(current_year)
   
   # Handle case where we need to catch up on previous year's data
   while year != current_year:
      year = get_recent_data(current_year)
   
   print(f"\n✓ Data extraction complete for {current_year}")

if __name__ == "__main__":
   main()
   