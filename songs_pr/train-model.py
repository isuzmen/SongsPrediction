import pandas as pd
import numpy as np

df = pd.read_csv('spotify-2023.csv', encoding='latin1')
df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
df['log_streams'] = np.log1p(df['streams'])