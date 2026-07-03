# auth.py - Complete Custom Authentication Module
import streamlit as st
import os
from pinecone import Pinecone
import bcrypt
import time
import jwt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import secrets

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
    if 'redirect_after_auth' not in st.session_state:
        st.session_state.redirect_after_auth = False

# ============================================
# PINECONE AUTH FUNCTIONS
# ============================================

def embed_text(text: str) -> list:
    """Simple embedding function"""
    import random
    random.seed(hash(text) % 2**32)
    return [random.uniform(0.01, 0.02) for _ in range(768)]

def find_user_by_email(email: str) -> dict:
    """Find user by email in Pinecone"""
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
    """Create new user in Pinecone"""
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
    """Verify user credentials"""
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
    """Generate JWT token"""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

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

# ============================================
# AUTHENTICATION CHECK WITH TIME GAP
# ============================================

def check_authentication():
    """
    Check if user is authenticated - uses query_params for persistence
    With time gap to handle async issues
    """
    init_session_state()
    
    # 1. Check session state first
    if st.session_state.logged_in and st.session_state.token:
        return True
    
    # 2. Check query params (survives page refresh)
    token_from_url = st.query_params.get("token", None)
    
    if token_from_url and not st.session_state.auth_checked:
        # Add small delay to ensure all async operations complete
        time.sleep(0.1)
        
        result = verify_token(token_from_url)
        if result.get('valid'):
            st.session_state.token = token_from_url
            st.session_state.user_id = result.get('user_id')
            st.session_state.user_email = result.get('email')
            st.session_state.logged_in = True
            st.session_state.auth_checked = True
            # Clear token from URL after validation
            st.query_params.clear()
            return True
        else:
            st.query_params.clear()
            st.session_state.auth_checked = True
    
    return False

def require_auth():
    """Require authentication for a page"""
    init_session_state()
    if not check_authentication():
        # Add delay before redirect to prevent race condition
        time.sleep(0.2)
        st.switch_page("app.py")
        st.stop()
    return True

def logout():
    """Logout user and clear all session data"""
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.logged_in = False
    st.session_state.auth_checked = False
    st.session_state.redirect_after_auth = False
    st.query_params.clear()