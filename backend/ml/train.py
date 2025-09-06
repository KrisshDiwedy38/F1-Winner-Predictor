import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from backend.data.cleaning import data_cleaning

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def create_advanced_features(df):
    """Create advanced features for F1 race prediction based on available data."""
    df = df.copy()
    
    print("Available columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    
    # Sort by race and position for proper feature engineering
    df = df.sort_values(['raceid', 'position']).reset_index(drop=True)
    
    # 1. DRIVER HISTORICAL PERFORMANCE
    print("Creating driver performance features...")
    
    # Calculate driver's recent win rate (last 10 races)
    df['driver_recent_wins'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y == 1).sum()).shift(1)
    ).fillna(0)
    
    # Driver's average finishing position (last 10 races)
    df['driver_avg_finish'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
    ).fillna(10)
    
    # Driver's podium rate (top 3 finishes in last 10 races)
    df['driver_podium_rate'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y <= 3).mean()).shift(1)
    ).fillna(0)
    
    # Driver's points finish rate (top 10 finishes)
    df['driver_points_rate'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y <= 10).mean()).shift(1)
    ).fillna(0)
    
    # Driver consistency (lower std = more consistent)
    df['driver_consistency'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).std().shift(1)
    ).fillna(5)
    df['driver_consistency'] = 1 / (1 + df['driver_consistency'])  # Invert so higher is better
    
    # 2. TEAM PERFORMANCE FEATURES
    print("Creating team performance features...")
    
    # Team's recent win rate (last 20 races)
    df['team_recent_wins'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y == 1).sum()).shift(1)
    ).fillna(0)
    
    # Team's average finishing position
    df['team_avg_finish'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).mean().shift(1)
    ).fillna(10)
    
    # Team's podium rate
    df['team_podium_rate'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y <= 3).mean()).shift(1)
    ).fillna(0)
    
    # Team's points rate
    df['team_points_rate'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y <= 10).mean()).shift(1)
    ).fillna(0)
    
    # 3. CIRCUIT-SPECIFIC PERFORMANCE
    print("Creating circuit-specific features...")
    
    # Driver performance at this specific circuit (historical)
    df['driver_circuit_wins'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: (x.shift(1) == 1).sum()
    ).fillna(0)
    
    df['driver_circuit_avg'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: x.shift(1).mean()
    ).fillna(10)
    
    df['driver_circuit_podiums'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: (x.shift(1) <= 3).sum()
    ).fillna(0)
    
    # Team performance at this circuit
    df['team_circuit_wins'] = df.groupby(['teamname', 'racename'])['position'].apply(
        lambda x: (x.shift(1) == 1).sum()
    ).fillna(0)
    
    df['team_circuit_avg'] = df.groupby(['teamname', 'racename'])['position'].apply(
        lambda x: x.shift(1).mean()
    ).fillna(10)
    
    # 4. WEATHER AND RACE CONDITIONS
    print("Creating weather and race condition features...")
    
    if 'rainfall' in df.columns:
        df['wet_race'] = (df['rainfall'] > 0).astype(int)
        df['heavy_rain'] = (df['rainfall'] > 2).astype(int)  # Moderate rain threshold
        df['light_rain'] = ((df['rainfall'] > 0) & (df['rainfall'] <= 2)).astype(int)
    else:
        df['wet_race'] = 0
        df['heavy_rain'] = 0
        df['light_rain'] = 0
    
    # 5. EXPERIENCE AND MOMENTUM FEATURES
    print("Creating experience and momentum features...")
    
    # Driver's total race count (experience) - simple cumulative count
    df['driver_experience'] = df.groupby('drivercode').cumcount() + 1
    
    # Initialize momentum features
    df['driver_recent_momentum'] = 10.0
    df['team_recent_momentum'] = 10.0
    
    # Recent momentum (last 5 races average position)
    for driver in df['drivercode'].unique():
        driver_mask = df['drivercode'] == driver
        driver_data = df[driver_mask].copy().sort_values('raceid')
        momentum = driver_data['position'].rolling(window=5, min_periods=1).mean().shift(1).fillna(10)
        df.loc[driver_mask, 'driver_recent_momentum'] = momentum.values
    
    # Team momentum (last 10 races)
    for team in df['teamname'].unique():
        team_mask = df['teamname'] == team
        team_data = df[team_mask].copy().sort_values('raceid')
        momentum = team_data['position'].rolling(window=10, min_periods=1).mean().shift(1).fillna(10)
        df.loc[team_mask, 'team_recent_momentum'] = momentum.values
    
    # 6. COMPETITIVE FEATURES
    print("Creating competitive features...")
    
    # Number of races this season
    df['season_race_number'] = df.groupby(['drivercode']).cumcount() + 1
    df['early_season'] = (df['season_race_number'] <= 6).astype(int)
    df['mid_season'] = ((df['season_race_number'] > 6) & (df['season_race_number'] <= 16)).astype(int)
    df['late_season'] = (df['season_race_number'] > 16).astype(int)
    
    # Driver vs team performance ratio
    df['driver_vs_team_ratio'] = df['driver_avg_finish'] / (df['team_avg_finish'] + 0.1)  # Avoid division by zero
    
    # 7. ENCODE CATEGORICAL VARIABLES
    print("Encoding categorical variables...")
    
    # Label encode drivers and teams for model use
    driver_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    circuit_encoder = LabelEncoder()
    
    df['drivercode_encoded'] = driver_encoder.fit_transform(df['drivercode'])
    df['teamname_encoded'] = team_encoder.fit_transform(df['teamname'])
    df['racename_encoded'] = circuit_encoder.fit_transform(df['racename'])
    
    # 8. NORMALIZE FEATURES (invert position-based features so higher is better)
    df['driver_avg_finish_norm'] = 21 - df['driver_avg_finish']
    df['team_avg_finish_norm'] = 21 - df['team_avg_finish']
    df['driver_circuit_avg_norm'] = 21 - df['driver_circuit_avg']
    df['team_circuit_avg_norm'] = 21 - df['team_circuit_avg']
    df['driver_recent_momentum_norm'] = 21 - df['driver_recent_momentum']
    df['team_recent_momentum_norm'] = 21 - df['team_recent_momentum']
    
    # Store encoders for later use
    encoders = {
        'driver_encoder': driver_encoder,
        'team_encoder': team_encoder,
        'circuit_encoder': circuit_encoder
    }
    
    print("Feature engineering completed!")
    print(f"Total features created: {len([col for col in df.columns if col not in ['position', 'raceid', 'racename', 'teamname', 'drivercode', 'fullname', 'rainfall', 'winner']])}")
    
    return df, encoders

def train_models():
    """Train the F1 race prediction model with advanced features."""
    print("Loading and cleaning data...")
    merged_df = data_cleaning()
    
    print("Creating advanced features...")
    df_with_features, encoders = create_advanced_features(merged_df)
    
    # Create target variable
    df_with_features['winner'] = (df_with_features['position'] == 1).astype(int)
    
    # Select the best features for prediction
    feature_cols = [
        # Driver performance features
        'driver_recent_wins', 'driver_avg_finish_norm', 'driver_podium_rate', 'driver_points_rate',
        'driver_consistency', 'driver_experience', 'driver_recent_momentum_norm', 'driver_vs_team_ratio',
        
        # Team performance features
        'team_recent_wins', 'team_avg_finish_norm', 'team_podium_rate', 'team_points_rate',
        'team_recent_momentum_norm',
        
        # Circuit-specific features
        'driver_circuit_wins', 'driver_circuit_avg_norm', 'driver_circuit_podiums',
        'team_circuit_wins', 'team_circuit_avg_norm',
        
        # Race conditions
        'wet_race', 'heavy_rain', 'light_rain',
        
        # Season and competitive features
        'early_season', 'mid_season', 'late_season',
        
        # Encoded categorical features
        'drivercode_encoded', 'teamname_encoded', 'racename_encoded'
    ]
    
    # Remove rows with missing target
    df_clean = df_with_features.dropna(subset=['winner']).copy()
    
    # Fill remaining NaN values with reasonable defaults
    for col in feature_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    print(f"Dataset shape after feature engineering: {df_clean.shape}")
    print(f"Features being used: {len(feature_cols)}")
    print(f"Winner distribution: \n{df_clean['winner'].value_counts()}")
    
    # Prepare features and target
    X = df_clean[feature_cols].copy()
    y = df_clean['winner'].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution - Winners: {y.sum()}, Non-winners: {len(y) - y.sum()}")
    
    # Remove any remaining infinite or very large values
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # XGBoost parameters optimized for imbalanced classification
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    print(f"Class imbalance ratio: {params['scale_pos_weight']:.2f}")
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("Training XGBoost model...")
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    # Evaluate model
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = model.get_score(importance_type='weight')
    print(f"\nTop 15 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    for feature, importance_val in sorted_importance:
        print(f"{feature}: {importance_val}")
    
    # Save model and metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'model_params': params,
        'feature_importance': importance,
        'accuracy': accuracy
    }
    
    model_path = os.path.join(MODEL_DIR, "race_predictor.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Features saved: {len(feature_cols)} features")
    
    return model_data

if __name__ == "__main__":
    train_models()