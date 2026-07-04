# auth.py - Authentication Module with Flask API
import streamlit as st
import os
import requests
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import uuid

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

API_URL = os.getenv("API_URL", "http://localhost:5000/api")

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
    if 'auth_checked' not in st.session_state:
        st.session_state.auth_checked = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None

# ============================================
# API CALL HELPER FUNCTIONS
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
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers)
        elif method == 'PUT':
            response = requests.put(url, json=data, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            return {'error': 'Invalid method'}
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            error_msg = response.json().get('error', 'API request failed')
            return {'error': error_msg}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to server. Please make sure the API is running.'}
    except Exception as e:
        return {'error': str(e)}

# ============================================
# SESSION STORE (Server-Side Persistence)
# ============================================

@st.cache_resource
def get_session_store():
    """
    A server-side session store that persists across reruns.
    This is the source of truth for all authenticated sessions.
    """
    return {}

def get_or_create_session_id():
    """
    Get or create a session ID.
    """
    init_session_state()
    
    # Check if we already have a session ID
    if st.session_state.session_id:
        return st.session_state.session_id
    
    # Create a new session ID
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id
    
    return session_id

# ============================================
# AUTHENTICATION FUNCTIONS
# ============================================

def check_authentication():
    """
    Check if user is authenticated.
    """
    init_session_state()
    
    # 1. Check session state first (fastest)
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # 2. Check with the API if we have a token
    if st.session_state.token:
        result = api_call('GET', 'auth/verify')
        if result.get('valid'):
            st.session_state.logged_in = True
            st.session_state.user_id = result.get('user_id')
            st.session_state.user_email = result.get('email')
            st.session_state.auth_checked = True
            return True
    
    return False

def login_user(email: str, password: str):
    """
    Login user via API.
    Returns (success, message)
    """
    init_session_state()
    
    # Validate input
    if not email or not password:
        return False, "Email and password are required"
    
    # Call login API
    result = api_call('POST', 'auth/login', {
        'email': email,
        'password': password
    })
    
    if 'error' in result:
        return False, result['error']
    
    # Set session state
    token = result.get('token')
    user_data = result.get('user', {})
    
    st.session_state.token = token
    st.session_state.user_id = user_data.get('user_id')
    st.session_state.user_email = user_data.get('email')
    st.session_state.logged_in = True
    st.session_state.auth_checked = True
    
    # Get or create session ID
    session_id = get_or_create_session_id()
    st.session_state.session_id = session_id
    
    # Store session data on the server
    session_store = get_session_store()
    session_store[session_id] = {
        'user_id': user_data.get('user_id'),
        'email': user_data.get('email'),
        'token': token,
        'login_time': datetime.now(timezone.utc).isoformat()
    }
    
    return True, "Login successful"

def register_user(email: str, password: str):
    """
    Register new user via API.
    Returns (success, message)
    """
    # Validate input
    if not email or not password:
        return False, "Email and password are required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    # Call register API
    result = api_call('POST', 'auth/register', {
        'email': email,
        'password': password
    })
    
    if 'error' in result:
        return False, result['error']
    
    return True, result.get('message', 'Registration successful')

def logout():
    """Logout user and clear all session data."""
    session_id = st.session_state.session_id
    if session_id:
        # Remove from server-side session store
        session_store = get_session_store()
        if session_id in session_store:
            del session_store[session_id]
    
    # Clear session state
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    st.session_state.auth_checked = False
    st.session_state.session_id = None
    
    # Force rerun to reflect changes
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

def is_authenticated():
    """Check if user is authenticated without side effects."""
    init_session_state()
    return st.session_state.logged_in and st.session_state.token is not None