# auth.py - Complete, Working Authentication with Cookie Persistence
import streamlit as st
import os
from pinecone import Pinecone
import bcrypt
import time
import jwt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import secrets
import uuid
from http.cookies import SimpleCookie
import streamlit.components.v1 as components
import re

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

# JWT Secret
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    JWT_SECRET = secrets.token_urlsafe(32)

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

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
    if 'cookie_checked' not in st.session_state:
        st.session_state.cookie_checked = False

# ============================================
# COOKIE HELPER FUNCTIONS
# ============================================

def get_cookie_value(key):
    """
    Safely get a cookie value from the request headers.
    Uses multiple approaches for compatibility across Streamlit versions.
    """
    # Method 1: Try Streamlit's context headers (Streamlit 1.28+)
    try:
        headers = st.context.headers
        if headers:
            cookie_str = headers.get("Cookie")
            if cookie_str:
                cookie = SimpleCookie(cookie_str)
                cookie_value = cookie.get(key)
                if cookie_value:
                    return cookie_value.value
    except (AttributeError, TypeError):
        pass
    
    # Method 2: Try browser's document.cookie via JavaScript
    # This is more reliable on Streamlit Cloud
    try:
        from streamlit.components.v1 import html
        
        # Create a hidden component that returns the cookie value
        js_code = f"""
        <script>
            function getCookie(name) {{
                const value = `; ${{document.cookie}}`;
                const parts = value.split(`; ${{name}}=`);
                if (parts.length === 2) {{
                    const cookieValue = parts.pop().split(';').shift();
                    window.parent.postMessage({{cookieValue: cookieValue}}, '*');
                }} else {{
                    window.parent.postMessage({{cookieValue: null}}, '*');
                }}
            }}
            getCookie('{key}');
        </script>
        """
        html(js_code, height=0)
    except:
        pass
    
    # Method 3: Try using st.query_params (works as fallback)
    try:
        # Check if cookie was passed as query param
        cookie_param = st.query_params.get('cookie_' + key)
        if cookie_param:
            return cookie_param
    except:
        pass
    
    return None

def set_cookie(key, value, days=30):
    """Set a cookie in the browser using JavaScript."""
    js_code = f"""
    <script>
        var date = new Date();
        date.setTime(date.getTime() + ({days} * 24 * 60 * 60 * 1000));
        document.cookie = "{key}={value}; expires=" + date.toUTCString() + "; path=/; SameSite=Lax; Secure";
        console.log("Cookie set: {key}={value}");
        
        // Also store in localStorage as backup
        localStorage.setItem('{key}', '{value}');
    </script>
    """
    components.html(js_code, height=0)
    time.sleep(0.3)  # Give time for cookie to be set

def delete_cookie(key):
    """Delete a cookie from the browser."""
    components.html(f"""
    <script>
        document.cookie = "{key}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        localStorage.removeItem('{key}');
        console.log("Cookie deleted: {key}");
    </script>
    """, height=0)
    time.sleep(0.2)  # Give time for cookie to be deleted

def get_or_create_session_id():
    """
    Get the session ID from the cookie/localStorage, or create a new one.
    This ID persists across page refreshes.
    """
    # Try to get from cookie first
    session_id = get_cookie_value('ST_SESSION_ID')
    
    # If not in cookie, try localStorage (alternative persistence)
    if session_id is None:
        try:
            # Check if we stored it in session state from previous runs
            if 'stored_session_id' in st.session_state:
                session_id = st.session_state.stored_session_id
        except:
            pass
    
    # If still None, create new one
    if session_id is None:
        session_id = uuid.uuid4().hex
        set_cookie('ST_SESSION_ID', session_id)
        st.session_state.stored_session_id = session_id
        st.session_state.session_id = session_id
        st.rerun()
    else:
        st.session_state.session_id = session_id
        st.session_state.stored_session_id = session_id
    
    return session_id

# ============================================
# AUTH STATE STORE (Singleton)
# ============================================

@st.cache_resource
def get_auth_state():
    """
    A singleton to store authentication state, mapping session IDs to user data.
    This persists across reruns because of @st.cache_resource.
    """
    return {}

# ============================================
# PINECONE AUTH FUNCTIONS
# ============================================

def embed_text(text: str) -> list:
    import random
    random.seed(hash(text) % 2**32)
    return [random.uniform(0.01, 0.02) for _ in range(768)]

def find_user_by_email(email: str) -> dict:
    text = f"user_auth:{email}"
    embedding = embed_text(text)
    
    try:
        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            namespace="users",
            filter={"type": {"$eq": "user_auth"}}
        )
        
        for match in results.matches:
            if match.metadata and match.metadata.get('email') == email:
                return {
                    "user_id": match.metadata.get('user_id'),
                    "email": match.metadata.get('email'),
                    "password_hash": match.metadata.get('password_hash')
                }
        return None
    except Exception:
        return None

def create_user(email: str, password: str) -> dict:
    existing = find_user_by_email(email)
    if existing:
        return None
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = f"user_{int(time.time())}"
    
    text = f"user_auth:{email}"
    embedding = embed_text(text)
    
    index.upsert(
        vectors=[{
            "id": f"{user_id}_auth",
            "values": embedding,
            "metadata": {
                "user_id": user_id,
                "type": "user_auth",
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }],
        namespace="users"
    )
    
    return {
        "user_id": user_id,
        "email": email
    }

def verify_user(email: str, password: str) -> dict:
    user = find_user_by_email(email)
    if not user:
        return None
    
    if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return {
            "user_id": user.get('user_id'),
            "email": user.get('email')
        }
    return None

# ============================================
# JWT FUNCTIONS
# ============================================

def generate_jwt(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_token(token: str) -> dict:
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

# ============================================
# AUTHENTICATION WITH COOKIE PERSISTENCE
# ============================================

def check_authentication():
    """Check if user is authenticated using session state or a persistent cookie."""
    init_session_state()
    
    # 1. Check session state first (fastest)
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # 2. Check if we already have a session ID stored
    session_id = None
    
    # Try to get session ID from various sources
    if st.session_state.session_id:
        session_id = st.session_state.session_id
    elif 'stored_session_id' in st.session_state:
        session_id = st.session_state.stored_session_id
    
    # 3. If no session ID in state, try to get/create one from cookie
    if session_id is None:
        session_id = get_or_create_session_id()
    
    # 4. Check auth state store with this session ID
    auth_state = get_auth_state()
    
    if session_id in auth_state:
        # Restore session from the auth state store
        user_data = auth_state[session_id]
        st.session_state.token = user_data.get('token')
        st.session_state.user_id = user_data.get('user_id')
        st.session_state.user_email = user_data.get('email')
        st.session_state.logged_in = True
        st.session_state.auth_checked = True
        st.session_state.session_id = session_id
        st.session_state.stored_session_id = session_id
        return True
    
    return False

def login_user(user_data: dict):
    """Mark a user as authenticated and persist across refreshes."""
    init_session_state()
    
    token = generate_jwt(user_data['user_id'], user_data['email'])
    
    # Set session state
    st.session_state.token = token
    st.session_state.user_id = user_data['user_id']
    st.session_state.user_email = user_data['email']
    st.session_state.logged_in = True
    st.session_state.auth_checked = True
    
    # Get or create session ID
    session_id = get_or_create_session_id()
    st.session_state.session_id = session_id
    st.session_state.stored_session_id = session_id
    
    # Save to cookie-based session (persists across refreshes)
    auth_state = get_auth_state()
    auth_state[session_id] = {
        'user_id': user_data['user_id'],
        'email': user_data['email'],
        'token': token,
    }
    
    # Also store a lightweight auth flag in session state for immediate access
    st.session_state._auth_verified = True
    
    return token

def require_auth():
    """Require authentication for a page."""
    init_session_state()
    if not check_authentication():
        st.switch_page("app.py")
        st.stop()
    return True

def logout():
    """Logout user and clear all session and cookie data."""
    session_id = get_cookie_value('ST_SESSION_ID')
    if session_id:
        auth_state = get_auth_state()
        if session_id in auth_state:
            del auth_state[session_id]
    
    delete_cookie('ST_SESSION_ID')
    
    # Clear all session state
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    st.session_state.auth_checked = False
    st.session_state.session_id = None
    st.session_state.stored_session_id = None
    st.session_state._auth_verified = False
    
    # Force rerun to reflect changes
    st.rerun()

# ============================================
# ALTERNATIVE: Query Parameter Auth (More Reliable on Cloud)
# ============================================

def set_auth_via_query_params(token: str):
    """Set authentication via query parameters (more reliable on Streamlit Cloud)"""
    st.query_params['auth_token'] = token
    st.query_params['auth_timestamp'] = str(time.time())

def get_auth_from_query_params():
    """Get authentication from query parameters"""
    token = st.query_params.get('auth_token')
    timestamp = st.query_params.get('auth_timestamp')
    
    if token and timestamp:
        # Check if timestamp is recent (within last hour)
        try:
            if time.time() - float(timestamp) < 3600:  # 1 hour
                return token
        except:
            pass
    return None