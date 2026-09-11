# app.py - Main Entry Point
import streamlit as st
from auth import check_authentication, init_session_state

init_session_state()

st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🤖",
    layout="centered"
)

# ✅ Check authentication from browser cookies
if check_authentication():
    st.switch_page("pages/Main_Page.py")
else:
    st.switch_page("pages/Authentication_Page.py")

st.stop()