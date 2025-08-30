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
   try:
      race_response = supabase.table("races").select("*").execute()
   except Exception as e:
      print(f"Error: {e}")
   else:
      race_data = race_response.data
      race_df = pd.DataFrame(race_data)

   try:
      weather_response = supabase.table("weather").select("*").execute()
   except Exception as e:
      print(f"Error: {e}")
   else:
      weather_data = weather_response.data
      weather_df = pd.DataFrame(weather_data)

   return race_df, weather_df



