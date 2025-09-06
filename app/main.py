import sys
import os

# Add project root (F1WinnerPredictor) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.data.cleaning import data_cleaning
from backend.ml.model import load_model, predict_race

st.title("🏎️ F1 Race Winner Predictor")

# Load cleaned dataset
merged_df = data_cleaning()

# Get unique race IDs and their corresponding names for selection
race_info = merged_df[["raceid", "racename"]].drop_duplicates()
race_options = {f"{row['racename']} (ID: {row['raceid']})": row['raceid'] 
                for _, row in race_info.iterrows()}

# Sidebar options
selected_race_display = st.sidebar.selectbox("Select a race", list(race_options.keys()))
selected_race_id = race_options[selected_race_display]

# For now, we only have one model (XGBoost), but keeping the structure for future models
model_choice = st.sidebar.radio(
    "Choose a model", ["XGBoost"]
)

# Show race information
race_data = merged_df[merged_df["raceid"] == selected_race_id]
if not race_data.empty:
    st.subheader(f"Race Information")
    st.write(f"**Race:** {race_data['racename'].iloc[0]}")
    st.write(f"**Race ID:** {selected_race_id}")
    st.write(f"**Number of drivers:** {len(race_data)}")
    
    # Show drivers in this race
    with st.expander("View all drivers in this race"):
        driver_info = race_data[["fullname", "teamname", "drivercode"]].drop_duplicates()
        st.dataframe(driver_info, use_container_width=True)

try:
    # Load model
    with st.spinner("Loading model..."):
        model_data = load_model()
    
    st.success("✅ Model loaded successfully!")
    
    # Show model information
    with st.expander("Model Information"):
        st.write(f"**Features used:** {', '.join(model_data['feature_cols'])}")
        st.write(f"**Number of target classes:** {len(model_data['encoders']['winner'].classes_)}")
    
    # Predict
    with st.spinner("Making predictions..."):
        try:
            results = predict_race(selected_race_id, model_data=model_data, top_n=5)
            
            st.subheader(f"🏁 Predicted Results for {race_data['racename'].iloc[0]} ({model_choice})")
            
            # Format the results for better display
            display_results = results.copy()
            display_results['win_probability'] = (display_results['pred_score'] * 100).round(1)
            display_results['rank'] = range(1, len(display_results) + 1)
            
            # Reorder columns for better display
            display_results = display_results[['rank', 'fullname', 'teamname', 'win_probability', 'pred_score']]
            display_results.columns = ['Rank', 'Driver', 'Team', 'Win Probability (%)', 'Raw Score']
            
            # Display results
            st.dataframe(display_results[['Rank', 'Driver', 'Team', 'Win Probability (%)']], 
                        use_container_width=True)
            
            # Highlight top prediction
            if not results.empty:
                top_driver = results.iloc[0]
                st.success(
                    f"🏆 **Predicted Winner:** {top_driver['fullname']} ({top_driver['teamname']}) "
                    f"with {(top_driver['pred_score'] * 100):.1f}% confidence score"
                )
                
                # Show top 3 as podium
                st.subheader("🥇 Predicted Podium")
                podium_cols = st.columns(3)
                
                podium_positions = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
                for i, (col, position) in enumerate(zip(podium_cols, podium_positions)):
                    if i < len(results):
                        driver = results.iloc[i]
                        with col:
                            st.metric(
                                label=position,
                                value=driver['fullname'],
                                delta=f"{driver['teamname']}"
                            )
            
        except ValueError as ve:
            st.error(f"❌ Prediction Error: {str(ve)}")
            st.info("This might happen if the race ID is not found in the dataset or if there are data issues.")
            
        except Exception as e:
            st.error(f"❌ Unexpected error during prediction: {str(e)}")
            
except FileNotFoundError:
    st.error("❌ **Model not found!** Please train the model first.")
    st.info("Run the following command in your terminal:")
    st.code("python backend/ml/train.py")
    
    # Option to train model from the app (if you want to add this feature)
    if st.button("🔧 Train Model Now"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                # Import and run training
                from backend.ml.train import train_models
                train_models()
                st.success("✅ Model trained successfully! Please refresh the page.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")

# Add footer with information
st.markdown("---")
st.markdown("### 📊 About this App")
st.markdown("""
- **Model:** XGBoost classifier trained on historical F1 race data
- **Features:** Rainfall conditions, driver performance, team performance
- **Prediction:** Based on historical patterns and race conditions
- **Note:** Predictions are for entertainment purposes and may not reflect actual race outcomes
""")