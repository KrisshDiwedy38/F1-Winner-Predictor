import os
import streamlit as st


try:
    from main import run_main_app
    from maintenance import maintenance_page

    # Get the app mode from an environment variable, defaulting to 'main'.
    APP_MODE = os.environ.get("APP_MODE", "main").lower()

    if APP_MODE == "maintenance":
        # If the mode is set to 'maintenance', show the maintenance page.
        maintenance_page()
    else:
        # Otherwise, show the main F1 predictor application.
        run_main_app()

except ImportError as e:
    st.set_page_config(page_title="Error", page_icon="🚨")
    st.title("🚨 Application Error")
    st.error(
        "A component of the application failed to load. "
        "This might be due to a missing file (`main.py` or `maintenance.py`) or a broken dependency."
    )
    st.code(f"Import Error: {e}")
