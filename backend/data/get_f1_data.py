import fastf1 as f1
import pandas as pd
import numpy as np

session = f1.get_session(2022,5,'R')
session.load()
result_df = session.results.copy()

print(result_df.columns)