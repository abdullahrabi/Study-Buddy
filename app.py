# app.py - Main Entry Point with Token Recovery
import streamlit as st
import time
from auth import check_authentication, init_session_state

# Initialize session state first
init_session_state()



st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🤖",
    layout="centered"
)

# Check authentication and route
# Add a small delay to allow cookie reading
time.sleep(1.0)

if check_authentication():
    st.switch_page("pages/Main_Page.py")
else:
    st.switch_page("pages/Authentication_Page.py")

st.stop()