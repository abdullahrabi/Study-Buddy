# Login_Signup.py - Fully Fixed
import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime, timedelta  # Removed timezone import
from dotenv import load_dotenv
from pinecone import Pinecone
import bcrypt
import jwt

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
# Use a strong key (at least 32 characters)
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-at-least-32-characters-long")

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="StudyBuddy - Login",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# SESSION STATE
# ============================================

if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ============================================
# REDIRECT TO MAIN APP IF LOGGED IN
# ============================================

if st.session_state.logged_in and st.session_state.token:
    # Check if app.py exists in the same directory
     st.switch_page("pages/app.py")
     st.stop()
   
# ============================================
# PINECONE SETUP
# ============================================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ============================================
# AUTH FUNCTIONS (Pinecone Only)
# ============================================

def embed_text(text: str) -> list:
    """Simple embedding function (use Gemini in production)"""
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
                "created_at": datetime.utcnow().isoformat()
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
        return user
    return None

def generate_jwt(user_id: str, email: str) -> str:
    """Generate JWT token"""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

# ============================================
# CSS STYLING
# ============================================

st.markdown("""
<style>
    /* Main container */
    .auth-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 40px 30px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf1;
    }
    
    /* Header */
    .auth-header {
        text-align: center;
        margin-bottom: 32px;
    }
    .auth-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .auth-header .emoji {
        font-size: 3rem;
        display: block;
        margin-bottom: 8px;
    }
    .auth-header .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    
    /* Form elements */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e5e7eb !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100% !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        background: #4f46e5 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f3f4f6;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Divider */
    .divider {
        display: flex;
        align-items: center;
        margin: 20px 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    .divider::before, .divider::after {
        content: "";
        flex: 1;
        border-bottom: 1px solid #e5e7eb;
    }
    .divider::before { margin-right: 16px; }
    .divider::after { margin-left: 16px; }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }
    
    /* Footer */
    .auth-footer {
        text-align: center;
        margin-top: 20px;
        color: #9ca3af;
        font-size: 0.8rem;
    }
    .auth-footer a {
        color: #6366f1;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOGIN / SIGNUP UI
# ============================================

# Header
st.markdown("""
<div class="auth-header">
    <span class="emoji">🤖</span>
    <h1>StudyBuddy</h1>
    <div class="subtitle">Your AI Study Assistant</div>
</div>
""", unsafe_allow_html=True)

# Container
with st.container():
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    # ============================================
    # LOGIN TAB
    # ============================================
    with tab1:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="login_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if not email or not password:
                    st.error("❌ Please fill in all fields.")
                else:
                    with st.spinner("Logging in..."):
                        user = verify_user(email, password)
                        if user:
                            token = generate_jwt(user['user_id'], user['email'])
                            st.session_state.token = token
                            st.session_state.user_id = user['user_id']
                            st.session_state.user_email = user['email']
                            st.session_state.logged_in = True
                            st.success(f"✅ Welcome back, {user['email']}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password.")
            
            # Divider
            st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
            
            # Quick demo note
            st.info("💡 Demo: Use 'demo@example.com' / 'password123'")
    
    # ============================================
    # SIGNUP TAB
    # ============================================
    with tab2:
        with st.form("signup_form", clear_on_submit=True):
            email = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="signup_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters",
                key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm your password",
                key="signup_confirm"
            )
            
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if not email or not password or not confirm_password:
                    st.error("❌ Please fill in all fields.")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match.")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account..."):
                        user = create_user(email, password)
                        if user:
                            token = generate_jwt(user['user_id'], user['email'])
                            st.session_state.token = token
                            st.session_state.user_id = user['user_id']
                            st.session_state.user_email = user['email']
                            st.session_state.logged_in = True
                            st.success(f"✅ Account created! Welcome, {user['email']}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ User with this email already exists.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="auth-footer">
    🔒 Secure authentication with JWT & Pinecone<br>
    <span style="color: #d1d5db;">v1.0</span>
</div>
""", unsafe_allow_html=True)