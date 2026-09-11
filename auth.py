# auth.py - Cookie-Based Authentication (Browser Cookies)
import streamlit as st
import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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

def set_browser_cookies(token: str, user_id: str, email: str):
    """Set cookies in the browser using JavaScript"""
    js_code = f"""
    <script>
        function setCookie(name, value, days) {{
            const expires = new Date();
            expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
            document.cookie = name + "=" + value + ";expires=" + expires.toUTCString() + ";path=/";
        }}
        setCookie("auth_token", "{token}", 7);
        setCookie("user_id", "{user_id}", 7);
        setCookie("user_email", "{email}", 7);
        console.log("✅ Browser cookies set!");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    time.sleep(0.5)  # Give time for cookies to set

def clear_browser_cookies():
    """Clear browser cookies using JavaScript"""
    js_code = """
    <script>
        document.cookie = "auth_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        document.cookie = "user_id=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        document.cookie = "user_email=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        console.log("✅ Browser cookies cleared!");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    time.sleep(0.5)

def get_browser_cookies():
    """Get cookies from browser using JavaScript"""
    js_code = """
    <script>
        function getCookie(name) {
            const value = `; ${document.cookie}`;
            const parts = value.split(`; ${name}=`);
            if (parts.length === 2) return parts.pop().split(';').shift();
            return null;
        }
        const token = getCookie('auth_token');
        const user_id = getCookie('user_id');
        const user_email = getCookie('user_email');
        window.parent.postMessage({
            type: 'auth_cookies',
            token: token,
            user_id: user_id,
            user_email: user_email
        }, '*');
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    time.sleep(0.5)

def check_authentication():
    """Check if user is authenticated from browser cookies"""
    init_session_state()
    
    # First check session state
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # Try to get cookies from browser
    get_browser_cookies()
    
    # Check if cookies were received
    if 'auth_token' in st.session_state:
        token = st.session_state.auth_token
        user_id = st.session_state.user_id_cookie
        user_email = st.session_state.user_email_cookie
        
        if token and user_id and user_email:
            # Verify token with backend
            try:
                response = requests.get(
                    f"{API_URL}/auth/verify",
                    params={'token': token},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get('valid'):
                        st.session_state.token = token
                        st.session_state.user_id = user_id
                        st.session_state.user_email = user_email
                        st.session_state.logged_in = True
                        return True
            except Exception as e:
                print(f"⚠️ Verification failed: {e}")
    
    return False

def login_user(email: str, password: str):
    """Login user and set browser cookies"""
    init_session_state()
    
    if not email or not password:
        return False, "Email and password are required"
    
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            json={'email': email, 'password': password},
            timeout=10
        )
        
        if response.status_code != 200:
            return False, response.json().get('error', 'Login failed')
        
        result = response.json()
        token = result.get('token')
        user_data = result.get('user', {})
        user_id = user_data.get('user_id')
        user_email = user_data.get('email')
        
        if token and user_id and user_email:
            # ✅ Set session state
            st.session_state.token = token
            st.session_state.user_id = user_id
            st.session_state.user_email = user_email
            st.session_state.logged_in = True
            
            # ✅ Set browser cookies (this is what you want!)
            set_browser_cookies(token, user_id, user_email)
            
            return True, "Login successful"
        
        return False, "Invalid response from server"
        
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def register_user(email: str, password: str):
    """Register new user"""
    if not email or not password:
        return False, "Email and password are required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    try:
        response = requests.post(
            f"{API_URL}/auth/register",
            json={'email': email, 'password': password},
            timeout=10
        )
        
        if response.status_code not in [200, 201]:
            return False, response.json().get('error', 'Registration failed')
        
        return True, "Registration successful"
        
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def logout():
    """Logout and clear everything"""
    # Clear session state
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    
    # Clear browser cookies
    clear_browser_cookies()
    
    st.rerun()

def require_auth():
    """Require authentication for a page"""
    init_session_state()
    if not check_authentication():
        st.switch_page("app.py")
        st.stop()
    return True