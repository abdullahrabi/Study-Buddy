# app.py - Main Entry Point with Token Recovery
import streamlit as st
import time
from auth import init_session_state, check_authentication, verify_token

st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🤖",
    layout="centered"
)

init_session_state()

# ============================================
# TRY TO RECOVER TOKEN FROM QUERY PARAMS
# ============================================

# Check if there's a token in the URL (from login redirect or refresh)
token_from_url = st.query_params.get("token", None)

if token_from_url and not st.session_state.logged_in:
    # Validate the token
    result = verify_token(token_from_url)
    if result.get('valid'):
        # Restore session state
        st.session_state.token = token_from_url
        st.session_state.user_id = result.get('user_id')
        st.session_state.user_email = result.get('email')
        st.session_state.logged_in = True
        st.session_state.auth_checked = True
        
        # Clear the token from URL
        st.query_params.clear()
        
        # Redirect to main page
        st.switch_page("pages/Main_Page.py")
        st.stop()
    else:
        # Invalid token, clear it
        st.query_params.clear()

# ============================================
# ROUTE TO APPROPRIATE PAGE
# ============================================

if check_authentication():
    st.switch_page("pages/Main_Page.py")
else:
    st.switch_page("pages/Authentication_Page.py")

st.stop()