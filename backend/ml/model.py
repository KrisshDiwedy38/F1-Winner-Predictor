import pandas as pd
from backend.data.cleaning import data_cleaning
import matplotlib.pyplot as plt
import seaborn as sns

clean_race_df , clean_weather_df = data_cleaning()


plt.figure(figsize=(20, 15))
plt.plot(
   clean_race_df[clean_race_df['drivercode'] == 'VER']['raceid'],
   clean_race_df[clean_race_df['drivercode'] == 'VER']['position'],
   marker='o', linestyle='-'
)

plt.title("Line Plot of HAM Times")
plt.xlabel("RaceID")
plt.ylabel("position")
plt.grid(True)
plt.show()