import fastf1 as f1
import pandas as pd
import sqlite3

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

# Converting values into a panads dataframe
f1_race_df = pd.DataFrame(result_values)
f1_weather_df = pd.DataFrame(weather_values)

# Storing Race data into SQL database : race_table
conn = sqlite3.connect("data\\race_data.db")
f1_race_df.to_sql("race_table", conn, if_exists="replace", index=False)
print("Race Data Stored Successfully")
conn.close()

# Storing Weather data into SQL database : weather_table
conn = sqlite3.connect("data\\weather_data.db")
f1_weather_df.to_sql("weather_table", conn, if_exists="replace", index = False)
print("Weather Data Stored Successfully")
conn.close()