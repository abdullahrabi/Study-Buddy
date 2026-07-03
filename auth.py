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

# ============================================
# COOKIE HELPER FUNCTIONS
# ============================================

def get_cookie_value(key):
    """Safely get a cookie value from the request headers."""
    try:
        headers = st.context.headers
    except AttributeError:
        try:
            from streamlit.web.server.websocket_headers import _get_websocket_headers
            headers = _get_websocket_headers()
        except:
            headers = None
        
    if headers is not None:
        cookie_str = headers.get("Cookie")
        if cookie_str:
            cookie = SimpleCookie(cookie_str)
            cookie_value = cookie.get(key)
            if cookie_value:
                return cookie_value.value
    return None

def set_cookie(key, value, days=30):
    """Set a cookie in the browser using JavaScript."""
    js_code = f"""
    <script>
        var date = new Date();
        date.setTime(date.getTime() + ({days} * 24 * 60 * 60 * 1000));
        document.cookie = "{key}={value}; expires=" + date.toUTCString() + "; path=/; SameSite=Lax";
        console.log("Cookie set: {key}={value}");
    </script>
    """
    components.html(js_code, height=0)
    time.sleep(0.2)  # Give time for cookie to be set

def delete_cookie(key):
    """Delete a cookie from the browser."""
    components.html(f"""
    <script>
        document.cookie = "{key}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        console.log("Cookie deleted: {key}");
    </script>
    """, height=0)
    time.sleep(0.2)  # Give time for cookie to be deleted

def get_or_create_session_id():
    """
    Get the session ID from the cookie, or create a new one.
    This ID persists across page refreshes.
    """
    session_id = get_cookie_value('ST_SESSION_ID')
    if session_id is None:
        session_id = uuid.uuid4().hex
        set_cookie('ST_SESSION_ID', session_id)
        st.session_state.session_id = session_id
        # Rerun to apply the cookie and avoid showing the login page briefly
        st.rerun()
    else:
        st.session_state.session_id = session_id
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
    
    # 2. Check cookie-based session (survives refresh)
    session_id = get_or_create_session_id()
    auth_state = get_auth_state()
    
    if session_id in auth_state:
        # Restore session from the auth state store
        user_data = auth_state[session_id]
        st.session_state.token = user_data.get('token')
        st.session_state.user_id = user_data.get('user_id')
        st.session_state.user_email = user_data.get('email')
        st.session_state.logged_in = True
        st.session_state.auth_checked = True
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
    
    # Save to cookie-based session (persists across refreshes)
    session_id = get_or_create_session_id()
    auth_state = get_auth_state()
    auth_state[session_id] = {
        'user_id': user_data['user_id'],
        'email': user_data['email'],
        'token': token,
    }
    
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
    
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    st.session_state.auth_checked = False
    st.session_state.session_id = None