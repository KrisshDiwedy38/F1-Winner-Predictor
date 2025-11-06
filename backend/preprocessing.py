import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sqlite3
import re
from datetime import datetime
import warnings

from update_data import get_recent_data
warnings.filterwarnings('ignore', category=FutureWarning)

current_year = datetime.now().year


# Load race data from SQLite database
conn = sqlite3.connect("data/race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

# Load weather data from SQLite database
conn = sqlite3.connect("data/weather_data.db")
weather_df = pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

def preprocessing():

   get_recent_data(current_year)
   # Handle missing Position values 
   max_position = race_df['Position'].max()
   race_df['Position'] = race_df['Position'].fillna(max_position + 1).astype(int)

   # Handle missing Time values (483 missing)

   # For finished drivers with missing times, estimate using interpolation
   for race_id in race_df['RaceID'].unique():
      race_mask = race_df['RaceID'] == race_id
      finished_mask = race_df['Status'] == 'Finished' 
      
      race_data = race_df[race_mask & finished_mask].copy()
      
      if race_data['Time(s)'].isnull().sum() > 0 and race_data['Time(s)'].notna().sum() > 1:
         race_data_sorted = race_data.sort_values('Position')
         race_data_sorted['Time(s)'] = race_data_sorted['Time(s)'].interpolate(method='linear')
         race_df.loc[race_mask & finished_mask, 'Time(s)'] = race_data_sorted['Time(s)'].values

   # Keep NaN for DNF drivers, but preserve times for Finished and Lapped drivers
   dnf_mask = ~race_df['Status'].isin(['Finished', 'Lapped'])
   race_df.loc[dnf_mask, 'Time(s)'] = np.nan

   # Handle "+X Lap(s)" status by adding time penalties
   lap_mask = race_df['Status'].str.contains(r'\+\d+\s+Lap', case=False, na=False)
   lap_indices = race_df[lap_mask].index

   for idx in lap_indices:
      status = race_df.loc[idx, 'Status']
      # Extract number of laps using regex
      match = re.search(r'\+(\d+)\s+Lap', status, re.IGNORECASE)
      if match:
         num_laps = int(match.group(1))
         penalty_seconds = num_laps * 100
         
         # Add penalty to existing time
         if pd.notna(race_df.loc[idx, 'Time(s)']):
            race_df.loc[idx, 'Time(s)'] += penalty_seconds
         else:
            # If no base time, just set the penalty
            race_df.loc[idx, 'Time(s)'] = penalty_seconds


   # Points given to the racer
   points_table = {
      1: 25,
      2: 18,
      3: 15,
      4: 12,
      5: 10,
      6: 8,
      7: 6,
      8: 4,
      9: 2,
      10: 1
   }

   race_df["Points"] = race_df["Position"].map(points_table).fillna(0).astype(int)
   
   # Create useful features
   race_df['Race_Year'] = race_df['RaceID'].str.extract(r'(\d{4})').astype(int)
   race_df['Race_Number'] = race_df['RaceID'].str.extract(r'_(\d+)').astype(int)
   race_df['Finished'] = (race_df['Status'] == 'Finished').astype(int)
   race_df['Points_Finish'] = (race_df['Position'] <= 10).astype(int)  # Top 10 get points
   race_df['Podium_Finish'] = (race_df['Position'] <= 3).astype(int)   # Top 3 get podium

   # Process weather data
   weather_df['Weather_Condition'] = weather_df['Rainfall'].map({True: 'Wet', False: 'Dry'})
   weather_df['Race_Year'] = weather_df['RaceID'].str.extract(r'(\d{4})').astype(int)
   weather_df['Race_Number'] = weather_df['RaceID'].str.extract(r'_(\d+)').astype(int)

   # Merge race and weather data
   merged_df = race_df.merge(weather_df, on='RaceID', how='left', suffixes=('', '_weather'))

   # Handle any missing weather data
   merged_df['Rainfall'] = merged_df['Rainfall'].fillna(False)
   merged_df['Weather_Condition'].fillna('Dry', inplace=True)

   # Remove duplicate columns from merge
   duplicate_cols = [col for col in merged_df.columns if col.endswith('_weather')]
   merged_df.drop(columns=duplicate_cols, inplace=True)

   # Remove duplicates and validate
   merged_df = merged_df.drop_duplicates(subset=['RaceID', 'DriverCode'])
   merged_df['Position'] = merged_df['Position'].abs()

   return merged_df

if __name__ == "__main__":
   result = preprocessing()
   print(result)