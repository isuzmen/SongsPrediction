import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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

numerical_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                     'acousticness_%', 'instrumentalness_%', 'liveness_%', 
                     'speechiness_%', 'encoded_key', 'encoded_mode',
                     'playlist_presence', 'chart_presence']

X = df[numerical_features].fillna(0)
y = df['log_streams'].fillna(df['log_streams'].mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)