import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from backend.data.cleaning import data_cleaning

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "race_predictor.pkl")

def create_advanced_features(df, encoders=None):
    """Create advanced features for F1 race prediction - same as training."""
    df = df.copy()
    
    # Sort by race and position for proper feature engineering
    df = df.sort_values(['raceid', 'position']).reset_index(drop=True)
    
    # 1. DRIVER HISTORICAL PERFORMANCE
    df['driver_recent_wins'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y == 1).sum()).shift(1)
    ).fillna(0)
    
    df['driver_avg_finish'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
    ).fillna(10)
    
    df['driver_podium_rate'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y <= 3).mean()).shift(1)
    ).fillna(0)
    
    df['driver_points_rate'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).apply(lambda y: (y <= 10).mean()).shift(1)
    ).fillna(0)
    
    df['driver_consistency'] = df.groupby('drivercode')['position'].apply(
        lambda x: x.rolling(window=10, min_periods=1).std().shift(1)
    ).fillna(5)
    df['driver_consistency'] = 1 / (1 + df['driver_consistency'])
    
    # 2. TEAM PERFORMANCE FEATURES
    df['team_recent_wins'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y == 1).sum()).shift(1)
    ).fillna(0)
    
    df['team_avg_finish'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).mean().shift(1)
    ).fillna(10)
    
    df['team_podium_rate'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y <= 3).mean()).shift(1)
    ).fillna(0)
    
    df['team_points_rate'] = df.groupby('teamname')['position'].apply(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: (y <= 10).mean()).shift(1)
    ).fillna(0)
    
    # 3. CIRCUIT-SPECIFIC PERFORMANCE
    df['driver_circuit_wins'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: (x.shift(1) == 1).sum()
    ).fillna(0)
    
    df['driver_circuit_avg'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: x.shift(1).mean()
    ).fillna(10)
    
    df['driver_circuit_podiums'] = df.groupby(['drivercode', 'racename'])['position'].apply(
        lambda x: (x.shift(1) <= 3).sum()
    ).fillna(0)
    
    df['team_circuit_wins'] = df.groupby(['teamname', 'racename'])['position'].apply(
        lambda x: (x.shift(1) == 1).sum()
    ).fillna(0)
    
    df['team_circuit_avg'] = df.groupby(['teamname', 'racename'])['position'].apply(
        lambda x: x.shift(1).mean()
    ).fillna(10)
    
    # 4. WEATHER AND RACE CONDITIONS
    if 'rainfall' in df.columns:
        df['wet_race'] = (df['rainfall'] > 0).astype(int)
        df['heavy_rain'] = (df['rainfall'] > 2).astype(int)
        df['light_rain'] = ((df['rainfall'] > 0) & (df['rainfall'] <= 2)).astype(int)
    else:
        df['wet_race'] = 0
        df['heavy_rain'] = 0
        df['light_rain'] = 0
    
    # 5. EXPERIENCE AND MOMENTUM FEATURES
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
    df['season_race_number'] = df.groupby(['drivercode']).cumcount() + 1
    df['early_season'] = (df['season_race_number'] <= 6).astype(int)
    df['mid_season'] = ((df['season_race_number'] > 6) & (df['season_race_number'] <= 16)).astype(int)
    df['late_season'] = (df['season_race_number'] > 16).astype(int)
    
    df['driver_vs_team_ratio'] = df['driver_avg_finish'] / (df['team_avg_finish'] + 0.1)
    
    # 7. ENCODE CATEGORICAL VARIABLES
    if encoders is not None:
        # Use existing encoders from training
        # Handle unseen categories by assigning a default value
        try:
            df['drivercode_encoded'] = df['drivercode'].apply(
                lambda x: encoders['driver_encoder'].transform([x])[0] if x in encoders['driver_encoder'].classes_ else -1
            )
            df['teamname_encoded'] = df['teamname'].apply(
                lambda x: encoders['team_encoder'].transform([x])[0] if x in encoders['team_encoder'].classes_ else -1
            )
            df['racename_encoded'] = df['racename'].apply(
                lambda x: encoders['circuit_encoder'].transform([x])[0] if x in encoders['circuit_encoder'].classes_ else -1
            )
        except Exception as e:
            print(f"Warning: Error in encoding: {e}")
            df['drivercode_encoded'] = 0
            df['teamname_encoded'] = 0
            df['racename_encoded'] = 0
    else:
        # Create new encoders (shouldn't happen during prediction)
        from sklearn.preprocessing import LabelEncoder
        df['drivercode_encoded'] = LabelEncoder().fit_transform(df['drivercode'])
        df['teamname_encoded'] = LabelEncoder().fit_transform(df['teamname'])
        df['racename_encoded'] = LabelEncoder().fit_transform(df['racename'])
    
    # 8. NORMALIZE FEATURES
    df['driver_avg_finish_norm'] = 21 - df['driver_avg_finish']
    df['team_avg_finish_norm'] = 21 - df['team_avg_finish']
    df['driver_circuit_avg_norm'] = 21 - df['driver_circuit_avg']
    df['team_circuit_avg_norm'] = 21 - df['team_circuit_avg']
    df['driver_recent_momentum_norm'] = 21 - df['driver_recent_momentum']
    df['team_recent_momentum_norm'] = 21 - df['team_recent_momentum']
    
    return df

def load_model(model_name="race_predictor.pkl"):
    """Load a trained model from disk."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    return model_data

def predict_race(race_id: str, model_data=None, top_n: int = 5):
    """
    Predict top drivers for a given race_id using advanced features.
    """
    if model_data is None:
        model_data = load_model()

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    encoders = model_data['encoders']

    print(f"Loading data for race {race_id}...")
    df = data_cleaning()
    
    # Create advanced features for the entire dataset
    df_with_features = create_advanced_features(df, encoders)
    
    # Filter for the specific race
    race_df = df_with_features[df_with_features["raceid"] == race_id].copy()
    
    if race_df.empty:
        raise ValueError(f"No data found for race_id {race_id}")
    
    print(f"Found {len(race_df)} drivers for race {race_id}")
    
    # Prepare features - fill any missing values
    for col in feature_cols:
        if col not in race_df.columns:
            print(f"Warning: Missing feature {col}, setting to 0")
            race_df[col] = 0
        else:
            race_df[col] = race_df[col].fillna(0)
    
    X_race = race_df[feature_cols].copy()
    
    # Clean data
    X_race = X_race.replace([np.inf, -np.inf], 0)
    X_race = X_race.fillna(0)
    
    print(f"Using {len(feature_cols)} features for prediction")
    
    # Make predictions
    dtest = xgb.DMatrix(X_race)
    win_probabilities = model.predict(dtest)
    
    # Add predictions to race data
    race_df["win_probability"] = win_probabilities
    race_df["pred_score"] = win_probabilities  # For compatibility
    
    # Sort by win probability and get top N
    top_results = race_df.sort_values("win_probability", ascending=False).head(top_n)
    
    # Prepare results
    results = top_results[["fullname", "teamname", "win_probability"]].copy()
    results["rank"] = range(1, len(results) + 1)
    
    print(f"Top {min(3, len(results))} predictions:")
    for i, (_, row) in enumerate(results.head(3).iterrows()):
        print(f"{i+1}. {row['fullname']} ({row['teamname']}) - {row['win_probability']:.3f}")
    
    return results[["fullname", "teamname", "win_probability"]]

def analyze_race_predictions(race_id: str, model_data=None):
    """
    Provide detailed analysis of race predictions including feature contributions.
    """
    if model_data is None:
        model_data = load_model()
    
    # Get predictions
    predictions = predict_race(race_id, model_data, top_n=10)
    
    # Get feature importance
    feature_importance = model_data.get('feature_importance', {})
    
    analysis = {
        'predictions': predictions,
        'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]),
        'model_accuracy': model_data.get('accuracy', 'Unknown')
    }
    
    return analysis

def get_model_info():
    """Get information about the loaded model."""
    try:
        model_data = load_model()
        return {
            "feature_columns": model_data['feature_cols'],
            "num_features": len(model_data['feature_cols']),
            "model_accuracy": model_data.get('accuracy', 'Unknown'),
            "top_features": dict(sorted(model_data.get('feature_importance', {}).items(), 
                               key=lambda x: x[1], reverse=True)[:10])
        }
    except Exception as e:
        return {"error": str(e)}