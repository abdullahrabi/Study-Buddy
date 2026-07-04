# app.py - Main Entry Point with Token Recovery
import streamlit as st
import time
from auth import check_authentication, init_session_state, setup_cookie_listener

# Initialize session state first
init_session_state()



st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🤖",
    layout="centered"
)
setup_cookie_listener()  # ✅ Now this exists
# Check authentication with query params
if check_authentication():
    st.switch_page("pages/Main_Page.py")
else:
    st.switch_page("pages/Authentication_Page.py")

st.stop()