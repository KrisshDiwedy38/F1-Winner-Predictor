import fastf1 as f1
import pandas as pd
import numpy as np
import sqlite3
from datetime import date
import os

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
   
   # Load race data from SQLite database
   conn = sqlite3.connect("data/race_data.db")
   h_race_df = pd.read_sql("SELECT * FROM race_table", conn)
   conn.close()
   
   result_values = []
   weather_values = []

   completed_races = completed_rounds(current_year)
   
   if completed_races == 0:
      print(f"No completed races found for {current_year}")
      return current_year

   # Get last race info from database
   last_race_id = h_race_df['RaceID'].iloc[-1] if not h_race_df.empty else "0_0"
   
   last_entry_year = int(last_race_id.split("_")[0])
   last_entry_race = int(last_race_id.split("_")[1])
   
   # Determine starting point for data extraction
   if last_entry_year == current_year and last_entry_race >= completed_races:    
      print("Data Already Up to Date")
      return current_year
   
   # Calculate starting race number
   if last_entry_year != current_year:
      last_season_races = completed_rounds(last_entry_year)
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
               
                        # Race Result table
            race_data = {
               'Position' : row.Position,
               'RaceID' : race_id,
               'RaceName' : race_name,
               'TeamName' : row.TeamName,
               'DriverCode' : row.Abbreviation,
               'FullName' : row.FullName,
               'Time(s)' : round(time,4),
               'Status' : row.Status
            }
            # Race Weather table
            weather_data = {
               'RaceID' : race_id,
               'RaceName' : race_name,
               'Rainfall' : rainfall
            }
            result_values.append(race_data)
         weather_values.append(weather_data)
         
         print(f"{race_id} - {race_name} completed")

   except Exception as e:
      print(f"Error occurred while fetching data: {e}")
      print(f"Failed at {current_year} Round {race_round}")
      return current_year

   # Convert to DataFrames
   f1_race_df = pd.DataFrame(result_values)
   f1_weather_df = pd.DataFrame(weather_values)

   # Storing Race data into SQL database : race_table
   conn = sqlite3.connect("data/f1_race_data.db")
   f1_race_df.to_sql("race_table", conn, if_exists="replace", index=False)
   print("Race Data Stored Successfully")
   conn.close()
   
   # Storing Weather data into SQL database : weather_table
   conn = sqlite3.connect("data/f1_weather_data.db")
   f1_weather_df.to_sql("weather_table", conn, if_exists="replace", index = False)
   print("Weather Data Stored Successfully")
   conn.close()

