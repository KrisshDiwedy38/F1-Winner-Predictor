from .get_supabase_data import load_data
import re
import pandas as pd
import numpy as np

race_df, weather_df = load_data()

def data_cleaning():
   # Keep NaN for DNF drivers, but preserve times for Finished and Lapped drivers
   dnf_mask = ~race_df['status'].isin(['Finished', 'Lapped'])
   race_df.loc[dnf_mask, 'timesecs'] = np.nan

   # Handle "+X Lap(s)" status by adding time penalties
   lap_mask = race_df['status'].str.contains(r'\+\d+\s+Lap', case=False, na=False)
   lap_indices = race_df[lap_mask].index

   for idx in lap_indices:
      status = race_df.loc[idx, 'status']
      # Extract number of laps using regex
      match = re.search(r'\+(\d+)\s+Lap', status, re.IGNORECASE)
      if match:
         num_laps = int(match.group(1))
         penalty_seconds = num_laps * 100
         
         # Add penalty to existing time
         if pd.notna(race_df.loc[idx, 'timesecs']):
            race_df.loc[idx, 'timesecs'] += penalty_seconds
         else:
            # If no base time, just set the penalty
            race_df.loc[idx, 'timesecs'] = penalty_seconds

   # Handling missing values in Weather DF 
   weather_df['rainfall'] = weather_df['rainfall'].fillna(0)

   merged_df = race_df.merge(weather_df[['raceid', 'rainfall']], on="raceid", how="left")

   merged_df["winner"] = (merged_df["position"] == 1).astype(int)

   return merged_df




