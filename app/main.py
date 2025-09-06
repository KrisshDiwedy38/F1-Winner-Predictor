import sys
import os
import streamlit as st

# Add project root (F1WinnerPredictor) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.cleaning import data_cleaning
from backend.ml.model import load_model, predict_race

def run_main_app():
    """
    The main function to run the F1 Winner Predictor Streamlit app.
    This function is called by app.py.
    """
    st.set_page_config(page_title="F1 Winner Predictor", page_icon="🏎️", layout="wide")
    st.title("🏎️ F1 Race Winner Predictor")

    # Load cleaned dataset
    merged_df = data_cleaning()

    # Get unique race IDs and names, sort by raceid to make the latest race the default
    race_info = merged_df[["raceid", "racename"]].drop_duplicates().sort_values("raceid", ascending=False)

    # Create an ordered list of options for the selectbox
    race_options_list = [f"{row['racename']} (ID: {row['raceid']})" for _, row in race_info.iterrows()]

    # Create a mapping from the display name back to the race ID
    race_options_map = {name: race_id for name, race_id in zip(race_options_list, race_info['raceid'])}

    # --- Sidebar Options ---
    st.sidebar.header("⚙️ Prediction Settings")

    # The selectbox will now default to the first item, which is the latest race
    selected_race_display = st.sidebar.selectbox("Select a Race", race_options_list)
    selected_race_id = race_options_map[selected_race_display]

    model_choice = st.sidebar.radio("Choose a Model", ["XGBoost"])

    # --- Main Page Content ---
    race_data = merged_df[merged_df["raceid"] == selected_race_id]
    if not race_data.empty:
        st.subheader(f"Race Information")
        st.write(f"**Race:** {race_data['racename'].iloc[0]}")
        st.write(f"**Race ID:** {selected_race_id}")
        st.write(f"**Number of drivers:** {len(race_data)}")

        with st.expander("View all drivers in this race"):
            driver_info = race_data[["fullname", "teamname", "drivercode"]].drop_duplicates()
            st.dataframe(driver_info, use_container_width=True)

    try:
        with st.spinner("Loading model..."):
            model_data = load_model()
        st.success("✅ Model loaded successfully!")

        with st.expander("Model Information"):
            st.write(f"**Features used:** {', '.join(model_data['feature_cols'])}")
            st.write(f"**Number of target classes:** {len(model_data['encoders']['winner'].classes_)}")

        with st.spinner("Making predictions..."):
            try:
                results = predict_race(selected_race_id, model_data=model_data, top_n=5)
                st.subheader(f"🏁 Predicted Results for {race_data['racename'].iloc[0]} ({model_choice})")

                display_results = results.copy()
                display_results['win_probability'] = (display_results['pred_score'] * 100).round(1)
                display_results['rank'] = range(1, len(display_results) + 1)
                display_results = display_results[['rank', 'fullname', 'teamname', 'win_probability']]
                display_results.columns = ['Rank', 'Driver', 'Team', 'Win Probability (%)']
                st.dataframe(display_results, use_container_width=True, hide_index=True)

                if not results.empty:
                    top_driver = results.iloc[0]
                    st.success(
                        f"🏆 **Predicted Winner:** {top_driver['fullname']} ({top_driver['teamname']}) "
                        f"with a **{(top_driver['pred_score'] * 100):.1f}%** confidence score."
                    )
                    st.subheader("🥇 Predicted Podium")
                    podium_cols = st.columns(3)
                    podium_positions = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
                    for i, (col, position) in enumerate(zip(podium_cols, podium_positions)):
                        if i < len(results):
                            driver = results.iloc[i]
                            with col:
                                st.metric(label=position, value=driver['fullname'], delta=f"{driver['teamname']}")

            except ValueError as ve:
                st.error(f"❌ Prediction Error: {str(ve)}")
            except Exception as e:
                st.error(f"❌ Unexpected error during prediction: {str(e)}")

    except FileNotFoundError:
        st.error("❌ **Model not found!** Please train the model first.")
        st.info("Run the following command in your terminal:")
        st.code("python backend/ml/train.py")
        if st.button("🔧 Train Model Now"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    from backend.ml.train import train_models
                    train_models()
                    st.success("✅ Model trained successfully! Please refresh the page.")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")

    st.markdown("---")
    st.markdown("### 📊 About this App")
    st.markdown("""
    - **Model:** XGBoost classifier trained on historical F1 race data.
    - **Features:** Grid position, qualifying performance, driver/team championship standings, and more.
    - **Prediction:** Probabilities are based on historical patterns and pre-race data.
    - **Note:** These predictions are for entertainment and educational purposes and may not reflect actual race outcomes.
    """)

# This allows the script to be run directly for testing purposes
if __name__ == "__main__":
    run_main_app()
