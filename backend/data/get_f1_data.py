import fastf1 as f1
import pandas as pd
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
      if session.weather_data["Rainfall"].any():
         rainfall = 1
      else:
         rainfall = 0
      
      # Getting required data
      result_data = session.results.loc[:,['TeamName','Abbreviation','FullName','Time', 'Position', 'Status']].copy()
      time = 0

      for row in result_data.sort_values("Position").itertuples():

         # Handling null time values given to lapped cars 
         if row.Status == "+1 Lap":
            time += 100.00
         elif row.Status == "+2 Laps":
            time += 200.00
         elif pd.isnull(row.Time):
            time += 0.0
         else:
            time += row.Time.total_seconds()

         
         # Race Result table
         race_data = {
            'Position' : row.Position,
            'RaceID' : race_id,
            'RaceName' : race_name,
            'TeamName' : row.TeamName,
            'DriverCode' : row.Abbreviation,
            'FullName' : row.FullName,
            'TimeSecs' : round(time,4),
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

race_dict = f1_result_df.to_dict(orient='records')
weather_dict = f1_weather_df.to_dict(orient='records')

batch_size = 500
for i in range(0, len(race_dict), batch_size):
    supabase.table('races').upsert(race_dict[i:i+batch_size]).execute()

for i in range(0, len(weather_dict), batch_size):
    supabase.table('weather').upsert(weather_dict[i:i+batch_size]).execute()

