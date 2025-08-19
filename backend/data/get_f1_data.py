import fastf1 as f1
import pandas as pd
import numpy as np
import streamlit as st

f1.Cache.enable_cache('backend\data\\f1cache')


years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
events = ['R']
rounds = [21, 21, 17, 22, 22 ,22 ,24]

result_values = []
weather_values = []

count = -1
for year in years:
   count += 1
   for event in events:
      for race_round in range(1,rounds[count] + 1):
         session = f1.get_session(year, race_round, event)
         session._load_weather_data()
         session._load_drivers_results()
         # Set a unique id for Primary Key and referencing other tables
         race_id = f"{str(year)}_{race_round}"
         # Extracting race name
         race_name = str(session).split(":")[1].split("-")[0].strip()
         # Checking if the race had rain
         if session.weather_data["Rainfall"].any():
            rainfall = "True"
         else:
            rainfall = "False"
         
         # Getting required data
         result_data = session.results.loc[:,['TeamName','Abbreviation','FullName','Time', 'Position', 'Status']].copy()
         time = 0

         for row in result_data.sort_values("Position").itertuples():

            # Handling null time values given to lapped cars 
            if row.Status == "+1 Lap" or row.Status == "+2 Laps":
               time += 10.00
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

# Converting values into a panads dataframe
f1_result_df = pd.DataFrame(result_values)
f1_weather_df = pd.DataFrame(weather_values)

f1_weather_df
f1_result_df