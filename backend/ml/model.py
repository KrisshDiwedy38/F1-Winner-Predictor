# ml/model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.get_supabase_data import load_data
race_df , weather_df = load_data()

if not race_df and not weather_df:
   print("Data not complete")
else:
   print(race_df)
   print(weather_df)