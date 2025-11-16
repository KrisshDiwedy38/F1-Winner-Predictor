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
conn = sqlite3.connect("C:/Users/HP/OneDrive/Desktop/Computer_Science/ResumeProjects/F1WinnerPredictor/data/results.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

# Load weather data from SQLite database
conn = sqlite3.connect("C:/Users/HP/OneDrive/Desktop/Computer_Science/ResumeProjects/F1WinnerPredictor/data/weather.db")
weather_df = pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

print(race_df)
print(weather_df)