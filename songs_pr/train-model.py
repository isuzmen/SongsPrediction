import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("\nCross-validation R² scores:", cv_scores)
print("Mean CV R² score:", cv_scores.mean())
print("CV score std:", cv_scores.std())

model.fit(X_train_scaled, y_train)

train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)
train_predictions_original = np.expm1(train_predictions)
test_predictions_original = np.expm1(test_predictions)
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

train_metrics = calculate_metrics(y_train_original, train_predictions_original)
test_metrics = calculate_metrics(y_test_original, test_predictions_original)

print("\nTraining Set Metrics:")
for metric, value in train_metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:,.2f}")

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:,.2f}")

feature_importance = pd.DataFrame({
    'feature': numerical_features,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Streams')
plt.ylabel('Predicted Log Streams')
plt.title('Actual vs Predicted Streams (Log Scale)')
plt.tight_layout()
plt.savefig('prediction_scatter.png')
plt.close()

print("\nTop 5 Most Important Features:")
print(feature_importance.head().to_string())

def plot_training_progress(model, X_train_scaled, y_train):
    n_estimators = model.n_estimators
    train_scores = []
    oob_scores = []
    
    for i in range(1, n_estimators + 1, 10):
        rf = RandomForestRegressor(
            n_estimators=i,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            random_state=42,
            oob_score=True
        )
        rf.fit(X_train_scaled, y_train)
        train_score = rf.score(X_train_scaled, y_train)
        oob_score = rf.oob_score_
        
        train_scores.append(train_score)
        oob_scores.append(oob_score)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    x_range = range(1, n_estimators + 1, 10)
    plt.plot(x_range, train_scores, label='Training R²', marker='o')
    plt.plot(x_range, oob_scores, label='Out-of-Bag R²', marker='s')
    plt.xlabel('Number of Trees')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Number of Trees')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores_cv, test_scores_cv = learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores_cv, axis=1)
    train_std = np.std(train_scores_cv, axis=1)
    test_mean = np.mean(test_scores_cv, axis=1)
    test_std = np.std(test_scores_cv, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
    plt.plot(train_sizes, test_mean, label='Cross-validation Score', marker='s')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.title('Learning Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

print("\nGenerating training progress plots...")
plot_training_progress(model, X_train_scaled, y_train)
print("Training progress plots have been saved as 'training_progress.png'")

plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
plt.scatter(y_test_original, test_predictions_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', lw=2)
plt.xlabel('Actual Streams')
plt.ylabel('Predicted Streams')
plt.title('Actual vs Predicted Streams')
plt.xscale('log')
plt.yscale('log')

plt.subplot(2, 2, 2)
feature_importance = pd.DataFrame({
    'feature': numerical_features,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')

plt.subplot(2, 2, 3)
residuals = y_test_original - test_predictions_original
plt.scatter(test_predictions_original, residuals, alpha=0.5)
plt.xlabel('Predicted Streams')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.xscale('log')
plt.axhline(y=0, color='r', linestyle='--')

plt.subplot(2, 2, 4)
sns.histplot(residuals, kde=True)
plt.xlabel('Residual Value')
plt.ylabel('Count')
plt.title('Distribution of Residuals')

plt.tight_layout()
plt.savefig('model_performance_plots.png')
plt.close()

print("\nVisualization plots have been saved as 'model_performance_plots.png'")
