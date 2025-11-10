import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocessing
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("F1 RACE PREDICTION SYSTEM")
print("="*80)
print("\nInitializing system...")


def load_data():
    """
    Load F1 race data from Lab4
    """

    df = preprocessing()
    return df

def preprocess_data(df):
    """
    Clean and prepare data for feature engineering
    """
    # Convert Position to numeric, handling DNFs
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    
    # Create a numeric finish position (DNFs get high value)
    df['FinishPosition'] = df['Position'].fillna(25)
    
    # Ensure Points is numeric
    df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    
    # Create binary flags
    df['DidFinish'] = df['Finished'].astype(int) if 'Finished' in df.columns else 1
    df['IsPodium'] = (df['Position'] <= 3).astype(int)
    df['IsWinner'] = (df['Position'] == 1).astype(int)
    df['PointsFinish'] = (df['Position'] <= 10).astype(int)
    
    return df


def calculate_rolling_features(df, lookback_races=5):
    """
    Calculate rolling performance features for each driver
    Features calculated:
    - Average finish position (last N races)
    - DNF rate (reliability)
    - Average points scored
    - Best finish in recent races
    - Championship position trend
    """
    df = df.sort_values(['DriverCode', 'Race_Year', 'Race_Number'])
    
    features_list = []
    
    for driver in df['DriverCode'].unique():
        driver_data = df[df['DriverCode'] == driver].copy()
        
        for idx in range(len(driver_data)):
            current_race = driver_data.iloc[idx]
            
            # Get previous races for this driver
            if idx < lookback_races:
                prev_races = driver_data.iloc[:idx]
            else:
                prev_races = driver_data.iloc[idx-lookback_races:idx]
            
            if len(prev_races) == 0:
                # New driver - use default values
                features = {
                    'avg_finish_position': 15.0,
                    'dnf_rate': 0.15,
                    'avg_points': 0.0,
                    'best_finish': 20.0,
                    'podium_rate': 0.0,
                    'races_experience': 0,
                    'points_rate': 0.0
                }
            else:
                features = {
                    'avg_finish_position': prev_races['FinishPosition'].mean(),
                    'dnf_rate': (1 - prev_races['DidFinish']).mean(),
                    'avg_points': prev_races['Points'].mean(),
                    'best_finish': prev_races['FinishPosition'].min(),
                    'podium_rate': prev_races['IsPodium'].mean(),
                    'races_experience': len(prev_races),
                    'points_rate': prev_races['PointsFinish'].mean()
                }
            
            # Add current race info
            features.update({
                'Position': current_race['Position'],
                'FinishPosition': current_race['FinishPosition'],
                'Points': current_race['Points'],
                'IsPodium': current_race['IsPodium'],
                'IsWinner': current_race['IsWinner'],
                'DriverCode': current_race['DriverCode'],
                'FullName': current_race['FullName'],
                'TeamName': current_race['TeamName'],
                'RaceName': current_race['RaceName'],
                'Race_Year': current_race['Race_Year'],
                'Race_Number': current_race['Race_Number']
            })
            
            features_list.append(features)
    
    return pd.DataFrame(features_list)


def calculate_team_features(df, features_df, lookback_races=5):
    """
    Add team performance features
    - Team's average points in recent races
    - Team's best finish
    - Team's overall form
    """
    df = df.sort_values(['Race_Year', 'Race_Number'])
    
    team_features = []
    
    for idx, row in features_df.iterrows():
        team = row['TeamName']
        year = row['Race_Year']
        race_num = row['Race_Number']
        
        # Get team's previous races
        team_history = df[
            (df['TeamName'] == team) & 
            ((df['Race_Year'] < year) | 
             ((df['Race_Year'] == year) & (df['Race_Number'] < race_num)))
        ].tail(lookback_races * 2)  # Consider both drivers
        
        if len(team_history) == 0:
            team_feat = {
                'team_avg_points': 0.0,
                'team_best_finish': 15.0,
                'team_podium_rate': 0.0
            }
        else:
            team_feat = {
                'team_avg_points': team_history['Points'].mean(),
                'team_best_finish': team_history['FinishPosition'].min(),
                'team_podium_rate': (team_history['Position'] <= 3).mean()
            }
        
        team_features.append(team_feat)
    
    team_df = pd.DataFrame(team_features)
    return pd.concat([features_df, team_df], axis=1)


def calculate_circuit_features(df, features_df):
    """
    Calculate circuit-specific performance for each driver
    - Average finish at this circuit
    - Best finish at this circuit
    - Times raced at this circuit
    """
    circuit_features = []
    
    for idx, row in features_df.iterrows():
        driver = row['DriverCode']
        circuit = row['RaceName']
        year = row['Race_Year']
        race_num = row['Race_Number']
        
        # Get driver's history at this circuit (before current race)
        circuit_history = df[
            (df['DriverCode'] == driver) & 
            (df['RaceName'] == circuit) &
            ((df['Race_Year'] < year) | 
             ((df['Race_Year'] == year) & (df['Race_Number'] < race_num)))
        ]
        
        if len(circuit_history) == 0:
            circuit_feat = {
                'circuit_avg_finish': 15.0,
                'circuit_best_finish': 20.0,
                'circuit_experience': 0
            }
        else:
            circuit_feat = {
                'circuit_avg_finish': circuit_history['FinishPosition'].mean(),
                'circuit_best_finish': circuit_history['FinishPosition'].min(),
                'circuit_experience': len(circuit_history)
            }
        
        circuit_features.append(circuit_feat)
    
    circuit_df = pd.DataFrame(circuit_features)
    return pd.concat([features_df, circuit_df], axis=1)


def train_models(features_df):
    """
    Train multiple ML models for different prediction tasks
    
    Models:
    1. Podium Predictor (Top 3 finish)
    2. Winner Predictor (1st place)
    3. Points Predictor (Top 10 finish)
    
    Using Gradient Boosting for better handling of feature interactions
    """
    # Select feature columns for training
    feature_columns = [
        'avg_finish_position',
        'dnf_rate',
        'avg_points',
        'best_finish',
        'podium_rate',
        'races_experience',
        'points_rate',
        'team_avg_points',
        'team_best_finish',
        'team_podium_rate',
        'circuit_avg_finish',
        'circuit_best_finish',
        'circuit_experience'
    ]
    
    X = features_df[feature_columns]
    
    # Remove rows with missing target values
    valid_data = features_df.dropna(subset=['IsPodium', 'IsWinner'])
    X = valid_data[feature_columns]
    
    # Train Podium Predictor
    print("\n" + "="*80)
    print("TRAINING PODIUM PREDICTOR")
    print("="*80)
    
    y_podium = valid_data['IsPodium']
    
    # Use Gradient Boosting for better performance
    podium_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    )
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_podium, test_size=0.2, random_state=42, stratify=y_podium
    )
    
    podium_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = podium_model.score(X_train, y_train)
    test_score = podium_model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_score:.3f}")
    print(f"Testing Accuracy: {test_score:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': podium_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # Train Winner Predictor
    print("\n" + "="*80)
    print("TRAINING WINNER PREDICTOR")
    print("="*80)
    
    y_winner = valid_data['IsWinner']
    
    winner_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    )
    
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X, y_winner, test_size=0.2, random_state=42, stratify=y_winner
    )
    
    winner_model.fit(X_train_w, y_train_w)
    
    print(f"Training Accuracy: {winner_model.score(X_train_w, y_train_w):.3f}")
    print(f"Testing Accuracy: {winner_model.score(X_test_w, y_test_w):.3f}")
    
    return podium_model, winner_model, feature_columns


def predict_race_outcome(models, feature_columns, features_df, race_year, race_name):
    """
    Predict outcome for a specific race
    """
    podium_model, winner_model = models
    
    # Get all drivers in this race
    race_data = features_df[
        (features_df['Race_Year'] == race_year) & 
        (features_df['RaceName'] == race_name)
    ].copy()
    
    if len(race_data) == 0:
        print(f"\nNo data found for {race_name} in {race_year}")
        return
    
    # Prepare features
    X_race = race_data[feature_columns]
    
    # Get predictions
    race_data['podium_prob'] = podium_model.predict_proba(X_race)[:, 1]
    race_data['winner_prob'] = winner_model.predict_proba(X_race)[:, 1]
    
    # Sort by combined score (weighted average)
    race_data['prediction_score'] = (
        race_data['winner_prob'] * 0.6 + 
        race_data['podium_prob'] * 0.4
    )
    
    race_data = race_data.sort_values('prediction_score', ascending=False)
    
    return race_data


def get_current_driver_features(df, driver, reference_date=None, lookback_races=5):
    """
    Calculate current features for a driver based on most recent races
    Used for future race predictions where we don't have the race data yet
    """
    if reference_date is None:
        # Use most recent data available
        driver_history = df[df['DriverCode'] == driver].copy()
    else:
        driver_history = df[
            (df['DriverCode'] == driver) & 
            (df['Race_Year'] <= reference_date['year'])
        ].copy()
    
    if len(driver_history) == 0:
        # Unknown driver - use neutral defaults
        return {
            'avg_finish_position': 15.0,
            'dnf_rate': 0.15,
            'avg_points': 0.0,
            'best_finish': 20.0,
            'podium_rate': 0.0,
            'races_experience': 0,
            'points_rate': 0.0
        }
    
    # Get last N races
    recent_races = driver_history.tail(lookback_races)
    
    features = {
        'avg_finish_position': recent_races['FinishPosition'].mean(),
        'dnf_rate': (1 - recent_races['DidFinish']).mean(),
        'avg_points': recent_races['Points'].mean(),
        'best_finish': recent_races['FinishPosition'].min(),
        'podium_rate': recent_races['IsPodium'].mean(),
        'races_experience': len(driver_history),
        'points_rate': recent_races['PointsFinish'].mean()
    }
    
    return features


def get_current_team_features(df, team, reference_date=None, lookback_races=5):
    """
    Calculate current team features based on most recent races
    """
    if reference_date is None:
        team_history = df[df['TeamName'] == team].copy()
    else:
        team_history = df[
            (df['TeamName'] == team) & 
            (df['Race_Year'] <= reference_date['year'])
        ].copy()
    
    if len(team_history) == 0:
        return {
            'team_avg_points': 0.0,
            'team_best_finish': 15.0,
            'team_podium_rate': 0.0
        }
    
    recent_races = team_history.tail(lookback_races * 2)  # Both drivers
    
    features = {
        'team_avg_points': recent_races['Points'].mean(),
        'team_best_finish': recent_races['FinishPosition'].min(),
        'team_podium_rate': (recent_races['Position'] <= 3).mean()
    }
    
    return features


def get_circuit_features(df, driver, circuit_name):
    """
    Get driver's historical performance at a specific circuit
    """
    circuit_history = df[
        (df['DriverCode'] == driver) & 
        (df['RaceName'].str.contains(circuit_name, case=False))
    ]
    
    if len(circuit_history) == 0:
        return {
            'circuit_avg_finish': 15.0,
            'circuit_best_finish': 20.0,
            'circuit_experience': 0
        }
    
    features = {
        'circuit_avg_finish': circuit_history['FinishPosition'].mean(),
        'circuit_best_finish': circuit_history['FinishPosition'].min(),
        'circuit_experience': len(circuit_history)
    }
    
    return features


def predict_future_race(models, feature_columns, df, race_name, year, drivers_teams):
    """
    Predict outcome for a future race that hasn't happened yet
    
    Args:
        models: Tuple of (podium_model, winner_model)
        feature_columns: List of feature column names
        df: Historical DataFrame with all past race data
        race_name: Name of the race (e.g., "Monaco Grand Prix")
        year: Year of the race
        drivers_teams: List of tuples [(driver_code, full_name, team_name), ...]
    """
    podium_model, winner_model = models
    
    print("\n" + "="*80)
    print(f"FUTURE RACE PREDICTION: {race_name} {year}")
    print("="*80)
    print("\nAnalyzing current form and historical data...")
    
    predictions = []
    
    for driver_code, full_name, team_name in drivers_teams:
        # Get current features for this driver
        driver_features = get_current_driver_features(df, driver_code)
        team_features = get_current_team_features(df, team_name)
        circuit_features = get_circuit_features(df, driver_code, race_name)
        
        # Combine all features
        feature_dict = {
            **driver_features,
            **team_features,
            **circuit_features
        }
        
        # Create feature vector
        feature_vector = pd.DataFrame([feature_dict])[feature_columns]
        
        # Get predictions
        podium_prob = podium_model.predict_proba(feature_vector)[0][1]
        winner_prob = winner_model.predict_proba(feature_vector)[0][1]
        
        # Calculate combined score
        prediction_score = winner_prob * 0.6 + podium_prob * 0.4
        
        predictions.append({
            'DriverCode': driver_code,
            'FullName': full_name,
            'TeamName': team_name,
            'winner_prob': winner_prob,
            'podium_prob': podium_prob,
            'prediction_score': prediction_score,
            **driver_features,
            **team_features,
            **circuit_features
        })
    
    # Convert to DataFrame and sort
    predictions_df = pd.DataFrame(predictions).sort_values(
        'prediction_score', ascending=False
    )
    
    # Display predictions
    print("\n" + "="*80)
    print("PREDICTED RACE RESULTS:")
    print("="*80)
    print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Win%':<8} {'Podium%':<10}")
    print("-"*80)
    
    for idx, (i, row) in enumerate(predictions_df.iterrows(), 1):
        driver = row['FullName']
        team = row['TeamName'][:24]
        win_prob = f"{row['winner_prob']*100:.1f}%"
        podium_prob = f"{row['podium_prob']*100:.1f}%"
        
        print(f"{idx:<5} {driver:<25} {team:<25} {win_prob:<8} {podium_prob:<10}")
    
    # Highlight predicted podium
    print("\n" + "="*80)
    print("🏆 PREDICTED PODIUM 🏆")
    print("="*80)
    
    for idx, (i, row) in enumerate(predictions_df.head(3).iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][idx-1]
        print(f"\n{medal} P{idx}: {row['FullName']}")
        print(f"   Team: {row['TeamName']}")
        print(f"   Win Probability: {row['winner_prob']*100:.1f}%")
        print(f"   Podium Probability: {row['podium_prob']*100:.1f}%")
        print(f"   Recent Form: Avg finish P{row['avg_finish_position']:.1f}, {row['avg_points']:.1f} pts/race")
        print(f"   Circuit Record: Avg P{row['circuit_avg_finish']:.1f} ({row['circuit_experience']} races)")
    
    print("\n" + "="*80 + "\n")
    
    return predictions_df


def display_race_analysis(race_data, race_name, race_year, show_actual=True):
    """
    Display race prediction and/or actual results
    """
    print("\n" + "="*80)
    print(f"RACE ANALYSIS: {race_name} {race_year}")
    print("="*80)
    
    if show_actual and 'Position' in race_data.columns:
        # Show actual results
        actual = race_data.sort_values('Position').head(10)
        print("\nACTUAL RESULTS:")
        print("-"*80)
        print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Points':<7} {'Status':<10}")
        print("-"*80)
        
        for idx, row in actual.iterrows():
            pos = int(row['Position']) if pd.notna(row['Position']) else 'DNF'
            driver = row['FullName']
            team = row['TeamName'][:24]
            points = int(row['Points'])
            status = 'Finished' if pd.notna(row['Position']) else 'DNF'
            
            print(f"{pos:<5} {driver:<25} {team:<25} {points:<7} {status:<10}")
    
    # Show predictions
    predicted = race_data.head(10)
    print("\n\nPREDICTED RESULTS:")
    print("-"*80)
    print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Win%':<8} {'Podium%':<10}")
    print("-"*80)
    
    for idx, (i, row) in enumerate(predicted.iterrows(), 1):
        driver = row['FullName']
        team = row['TeamName'][:24]
        win_prob = f"{row['winner_prob']*100:.1f}%"
        podium_prob = f"{row['podium_prob']*100:.1f}%"
        
        print(f"{idx:<5} {driver:<25} {team:<25} {win_prob:<8} {podium_prob:<10}")
    
    # Highlight predicted podium
    print("\n" + "="*80)
    print("PREDICTED PODIUM:")
    print("="*80)
    for idx, (i, row) in enumerate(predicted.head(3).iterrows(), 1):
        print(f"\n{idx}. {row['FullName']} - {row['TeamName']}")
        print(f"   Win Probability: {row['winner_prob']*100:.1f}%")
        print(f"   Podium Probability: {row['podium_prob']*100:.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main execution flow
    """
    # Load data
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    # Feature engineering
    print("Calculating driver performance features...")
    features_df = calculate_rolling_features(df, lookback_races=5)
    
    print("Calculating team performance features...")
    features_df = calculate_team_features(df, features_df, lookback_races=5)
    
    print("Calculating circuit-specific features...")
    features_df = calculate_circuit_features(df, features_df)
    
    print(f"\n✓ Feature engineering complete: {len(features_df)} entries with {len(features_df.columns)} features")
    
    # Train models
    podium_model, winner_model, feature_columns = train_models(features_df)
    models = (podium_model, winner_model)
    
    # Interactive prediction
    print("\n" + "="*80)
    print("PREDICTION SYSTEM READY")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Predict/Analyze a specific race (from historical data)")
        print("2. Predict a FUTURE race (custom driver lineup)")
        print("3. Show available races in dataset")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            race_name = input("\nEnter race name (e.g., 'Australian Grand Prix'): ").strip()
            year_input = input("Enter year: ").strip()
            
            try:
                year = int(year_input)
            except:
                print("Invalid year")
                continue
            
            # Find matching races
            matches = features_df[
                (features_df['RaceName'].str.contains(race_name, case=False)) &
                (features_df['Race_Year'] == year)
            ]
            
            if len(matches) == 0:
                print(f"\nNo matches found for '{race_name}' in {year}")
                continue
            
            race_name_exact = matches.iloc[0]['RaceName']
            
            # Make prediction
            race_data = predict_race_outcome(
                models, feature_columns, features_df, year, race_name_exact
            )
            
            if race_data is not None:
                display_race_analysis(race_data, race_name_exact, year, show_actual=True)
        
        elif choice == '2':
            print("\n" + "="*80)
            print("FUTURE RACE PREDICTION")
            print("="*80)
            
            race_name = input("\nEnter race name (e.g., 'Monaco Grand Prix', 'Las Vegas Grand Prix'): ").strip()
            year_input = input("Enter year: ").strip()
            
            try:
                year = int(year_input)
            except:
                print("Invalid year")
                continue
            
            print("\nNow enter the driver lineup for this race.")
            print("You can either:")
            print("  A) Use current grid (auto-detect from latest data)")
            print("  B) Manually enter drivers")
            
            lineup_choice = input("\nChoose A or B: ").strip().upper()
            
            if lineup_choice == 'A':
                # Auto-detect current grid from most recent data
                latest_year = df['Race_Year'].max()
                latest_races = df[df['Race_Year'] == latest_year]
                
                # Get unique drivers from latest season
                current_grid = latest_races.groupby('DriverCode').agg({
                    'FullName': 'first',
                    'TeamName': 'last'  # Most recent team
                }).reset_index()
                
                print(f"\n✓ Detected {len(current_grid)} drivers from {latest_year} season:")
                for idx, row in current_grid.iterrows():
                    print(f"  {row['DriverCode']}: {row['FullName']} ({row['TeamName']})")
                
                drivers_teams = [
                    (row['DriverCode'], row['FullName'], row['TeamName']) 
                    for idx, row in current_grid.iterrows()
                ]
                
            else:  # Manual entry
                print("\nEnter drivers one at a time (press Enter with empty driver code to finish):")
                drivers_teams = []
                
                while True:
                    print(f"\nDriver #{len(drivers_teams) + 1}:")
                    driver_code = input("  Driver Code (3 letters, e.g., VER, HAM): ").strip().upper()
                    
                    if not driver_code:
                        if len(drivers_teams) >= 10:  # At least 10 drivers
                            break
                        else:
                            print("  Please enter at least 10 drivers")
                            continue
                    
                    full_name = input("  Full Name: ").strip()
                    team_name = input("  Team Name: ").strip()
                    
                    drivers_teams.append((driver_code, full_name, team_name))
                    
                    if len(drivers_teams) >= 20:  # F1 grid typically has 20 drivers
                        break
            
            # Make prediction for future race
            predict_future_race(models, feature_columns, df, race_name, year, drivers_teams)
        
        elif choice == '3':
            print("\nAvailable Races:")
            print("-"*80)
            races = features_df.groupby(['Race_Year', 'RaceName']).size().reset_index()
            for idx, row in races.iterrows():
                print(f"{row['Race_Year']} - {row['RaceName']}")
        
        elif choice == '4':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()