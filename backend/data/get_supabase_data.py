import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase : Client = create_client(supabase_url,supabase_key)

def load_data():
   race_df, weather_df = None, None

   # Fetch races
   try:
      all_races = []
      current_page = 0
      page_size = 1000
      while True:
         start_index = current_page * page_size
         end_index = start_index + page_size - 1

         race_response = supabase.table("races").select("*").range(start_index, end_index).execute()
         all_races.extend(race_response.data)

         if len(race_response.data) < page_size:
               break
         current_page += 1

      race_df = pd.DataFrame(all_races)

   except Exception as e:
      print(f"Error loading races: {e}")

   # Fetch weather
   try:
      weather_response = supabase.table("weather").select("*").execute()
      weather_df = pd.DataFrame(weather_response.data)
   except Exception as e:
      print(f"Error loading weather: {e}")

   return race_df, weather_df




