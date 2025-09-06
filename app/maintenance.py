import streamlit as st

# Set the page configuration for a cleaner look
st.set_page_config(
    page_title="Under Maintenance",
    page_icon="🛠️",
    layout="centered"
)

def maintenance_page():
    """
    Displays a user-friendly maintenance page for the Streamlit app.
    """
    st.title("🛠️ Application Under Maintenance")
    
    st.warning("We're currently performing some essential updates and improvements.")
    
    st.markdown("""
    ---
    
    ### We'll be back shortly!
    
    We apologize for any inconvenience this may cause. Our team is working hard to get things back up and running as quickly as possible.
    
    Thank you for your patience.
    
    *— The Development Team*
    """)
    
    st.info("Please check back in a little while.")

if __name__ == "__main__":
    maintenance_page()
