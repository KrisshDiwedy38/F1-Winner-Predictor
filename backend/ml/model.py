import pandas as pd
from backend.data.get_supabase_data import load_data 

race_df , weather_df = load_data()

print(race_df)
print(weather_df)