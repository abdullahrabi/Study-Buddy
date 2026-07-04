# app.py - Main Entry Point with Token Recovery
import streamlit as st
import time
from auth import check_authentication

st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# ROUTE TO APPROPRIATE PAGE
# ============================================

if check_authentication():
    st.switch_page("pages/Main_Page.py")
else:
    st.switch_page("pages/Authentication_Page.py")

st.stop()