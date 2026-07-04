# auth.py - Authentication Module with Query Params Only (Simplest & Most Reliable)
import streamlit as st
import os
import requests
import json
import base64
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

API_URL = os.getenv("API_URL")

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize all session state variables"""
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

# ============================================
# API CALL HELPERS
# ============================================

def get_headers():
    """Get headers with auth token"""
    headers = {'Content-Type': 'application/json'}
    if st.session_state.token:
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    return headers

def api_call(method: str, endpoint: str, data: dict = None, params: dict = None):
    """Generic API call function"""
    url = f"{API_URL}/{endpoint}"
    headers = get_headers()
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            return {'error': 'Invalid method'}
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            error_msg = response.json().get('error', 'API request failed')
            return {'error': error_msg}
    except Exception as e:
        return {'error': str(e)}

# ============================================
# AUTHENTICATION FUNCTIONS
# ============================================

def check_authentication():
    """
    Check if user is authenticated.
    Restores from query_params if session state is empty.
    """
    init_session_state()
    
    # Check session state first
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # Try to restore from query_params
    token = st.query_params.get('token')
    user_id = st.query_params.get('user_id')
    user_email = st.query_params.get('user_email')
    
    if token and user_id and user_email:
        # Verify token with backend
        st.session_state.token = token
        result = api_call('GET', 'auth/verify')
        
        if result.get('valid'):
            st.session_state.user_id = user_id
            st.session_state.user_email = user_email
            st.session_state.logged_in = True
            return True
        else:
            # Clear invalid query params
            st.query_params.clear()
    
    return False

def login_user(email: str, password: str):
    """
    Login user via API.
    Returns (success, message)
    """
    init_session_state()
    
    if not email or not password:
        return False, "Email and password are required"
    
    # Call login API
    result = api_call('POST', 'auth/login', {
        'email': email,
        'password': password
    })
    
    if 'error' in result:
        return False, result['error']
    
    # Get data from response
    token = result.get('token')
    user_data = result.get('user', {})
    user_id = user_data.get('user_id')
    user_email = user_data.get('email')
    
    if not token or not user_id or not user_email:
        return False, "Invalid response from server"
    
    # Set session state
    st.session_state.token = token
    st.session_state.user_id = user_id
    st.session_state.user_email = user_email
    st.session_state.logged_in = True
    
    # Save to query_params for persistence across refreshes
    st.query_params['token'] = token
    st.query_params['user_id'] = user_id
    st.query_params['user_email'] = user_email
    
    return True, "Login successful"

def register_user(email: str, password: str):
    """
    Register new user via API.
    Returns (success, message)
    """
    if not email or not password:
        return False, "Email and password are required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    result = api_call('POST', 'auth/register', {
        'email': email,
        'password': password
    })
    
    if 'error' in result:
        return False, result['error']
    
    return True, result.get('message', 'Registration successful')

def logout():
    """Logout user and clear all session data."""
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    
    # Clear query params
    st.query_params.clear()
    
    st.rerun()

def require_auth():
    """Require authentication for a page."""
    init_session_state()
    if not check_authentication():
        st.switch_page("app.py")
        st.stop()
    return True

def get_current_user():
    """Get the current authenticated user's data."""
    if check_authentication():
        return {
            'user_id': st.session_state.user_id,
            'email': st.session_state.user_email
        }
    return None