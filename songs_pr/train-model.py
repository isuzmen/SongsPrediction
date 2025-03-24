import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('spotify-2023.csv', encoding='latin1')
df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
df['log_streams'] = np.log1p(df['streams'])

le_key = LabelEncoder()
le_mode = LabelEncoder()
df['encoded_key'] = le_key.fit_transform(df['key'].fillna('Unknown'))
df['encoded_mode'] = le_mode.fit_transform(df['mode'])
df['playlist_presence'] = df['in_spotify_playlists'] + df['in_apple_playlists']
df['chart_presence'] = df['in_spotify_charts'] + df['in_apple_charts'] + df['in_deezer_charts']