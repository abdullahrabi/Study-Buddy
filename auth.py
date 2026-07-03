# auth.py - Shared authentication module
import streamlit as st
import jwt
from datetime import datetime, timezone, timedelta
import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    JWT_SECRET = secrets.token_urlsafe(32)

def init_session_state():
    """Initialize session state variables"""
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

def verify_token(token: str) -> dict:
    """Verify JWT token and return user data if valid"""
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        exp = decoded.get('exp')
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) > datetime.now(timezone.utc):
            return {
                'user_id': decoded.get('user_id'),
                'email': decoded.get('email'),
                'valid': True
            }
        return {'valid': False}
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return {'valid': False}

def check_authentication():
    """Check if user is authenticated"""
    init_session_state()
    
    # If already logged in session, return True
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # Check for token in query params (from localStorage redirect)
    query_params = st.query_params
    token_from_url = query_params.get("token", None)
    
    if token_from_url and not st.session_state.auth_checked:
        result = verify_token(token_from_url)
        if result.get('valid'):
            st.session_state.token = token_from_url
            st.session_state.user_id = result.get('user_id')
            st.session_state.user_email = result.get('email')
            st.session_state.logged_in = True
            st.session_state.auth_checked = True
            st.query_params.clear()
            return True
        else:
            st.query_params.clear()
    
    return False

def require_auth():
    """Require authentication for a page"""
    init_session_state()
    if not check_authentication():
        st.switch_page("app.py")
        st.stop()
    return True

def logout():
    """Logout user and clear session"""
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    st.session_state.auth_checked = False
    # Clear localStorage via JavaScript
    import streamlit.components.v1 as components
    components.html("""
    <script>
        localStorage.removeItem('auth_token');
        localStorage.removeItem('token_timestamp');
        window.location.href = window.location.pathname;
    </script>
    """, height=0)