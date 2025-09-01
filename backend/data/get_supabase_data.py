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
      all_races = []
      current_page = 0
      page_size = 1000
      while True:
         # Calculate the range for the current page
         start_index = current_page * page_size
         end_index = start_index + page_size - 1

         # Fetch a "page" of data
         race_response = supabase.table("races").select("*").range(start_index, end_index).execute()

         # Add the fetched data to our list
         all_races.extend(race_response.data)

         # If the number of returned rows is less than the page size,
         # it means we have reached the last page.
         if len(race_response.data) < page_size:
            break

         # Move to the next page
         current_page += 1
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



