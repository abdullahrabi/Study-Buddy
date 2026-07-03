# Login_Signup.py - Fully Fixed with Local Storage
import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pinecone import Pinecone
import bcrypt
import jwt
import streamlit.components.v1 as components

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
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
# JAVASCRIPT FOR LOCAL STORAGE MANAGEMENT
# ============================================

def get_local_storage_js():
    """JavaScript to handle local storage operations"""
    return """
    <script>
    // Function to set token in local storage
    function setToken(token) {
        if (token) {
            localStorage.setItem('auth_token', token);
            localStorage.setItem('token_timestamp', Date.now().toString());
            return true;
        }
        return false;
    }
    
    // Function to get token from local storage
    function getToken() {
        return localStorage.getItem('auth_token');
    }
    
    // Function to remove token from local storage
    function removeToken() {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('token_timestamp');
    }
    
    // Function to check if token is expired
    function isTokenExpired() {
        const timestamp = localStorage.getItem('token_timestamp');
        if (!timestamp) return true;
        const now = Date.now();
        const expiryTime = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds
        return (now - parseInt(timestamp)) > expiryTime;
    }
    
    // Check token on page load and send to Streamlit
    window.onload = function() {
        const token = getToken();
        const expired = isTokenExpired();
        if (token && !expired) {
            // Send token to Streamlit backend
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: token
            }, '*');
        } else if (expired) {
            removeToken();
        }
    };
    </script>
    """

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
if 'token_from_storage' not in st.session_state:
    st.session_state.token_from_storage = None

# ============================================
# CHECK LOCAL STORAGE FOR TOKEN
# ============================================

# Embed JavaScript to read from local storage
components.html(get_local_storage_js(), height=0)

# Try to get token from local storage via query params
# This is a workaround since Streamlit doesn't directly access localStorage
query_params = st.query_params
token_from_url = query_params.get("token", None)

if token_from_url:
    # Validate the token
    try:
        decoded = jwt.decode(token_from_url, JWT_SECRET, algorithms=['HS256'])
        # Check if token is expired
        exp = decoded.get('exp')
        if exp and datetime.utcfromtimestamp(exp) > datetime.utcnow():
            st.session_state.token = token_from_url
            st.session_state.user_id = decoded.get('user_id')
            st.session_state.user_email = decoded.get('email')
            st.session_state.logged_in = True
            # Clear the query params after setting session
            st.query_params.clear()
    except jwt.ExpiredSignatureError:
        pass
    except jwt.InvalidTokenError:
        pass

# ============================================
# LOAD CSS
# ============================================

def load_css():
    """Load CSS from external file"""
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background: linear-gradient(180deg, #061022, #0b1523) !important;
                color: #e6eef9 !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px;
                margin-bottom: -10px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 8px 16px;
                margin: 0px;
            }
            .stTabs [role="tabpanel"] {
                padding-top: 5px;
            }
            .stForm {
                margin-top: 0px;
                padding-top: 0px;
            }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ============================================
# REDIRECT TO MAIN APP IF LOGGED IN
# ============================================

if st.session_state.logged_in and st.session_state.token:
    st.switch_page("pages/main.py")
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
# JAVASCRIPT TO SET TOKEN IN LOCAL STORAGE
# ============================================

def set_token_in_storage(token):
    """Inject JavaScript to save token to localStorage"""
    js_code = f"""
    <script>
        localStorage.setItem('auth_token', '{token}');
        localStorage.setItem('token_timestamp', Date.now().toString());
        console.log('Token saved to localStorage');
    </script>
    """
    components.html(js_code, height=0)

# ============================================
# LOGIN / SIGNUP UI
# ============================================

# Header Section
st.markdown(
    """
    <div class="header-card">
        <div class="header-inner">
            <div class="header-avatar">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
            </div>
            <div>
                <h1>Meet StudyBuddy</h1>
                <p>Your Personal AI Learning Assistant with Advanced Quiz Features</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs Section
tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])

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
        
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not email or not password:
                st.toast("❌ Please fill in all fields.")
            else:
                with st.spinner("Logging in..."):
                    user = verify_user(email, password)
                    if user:
                        token = generate_jwt(user['user_id'], user['email'])
                        
                        # Store token in session state
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        
                        # Save token to localStorage
                        set_token_in_storage(token)
                        
                        st.toast(f"✅ Welcome back, {user['email']}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.toast("❌ Invalid email or password.")

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
        
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            if not email or not password or not confirm_password:
                st.toast("❌ Please fill in all fields.")
            elif password != confirm_password:
                st.toast("❌ Passwords do not match.")
            elif len(password) < 6:
                st.toast("❌ Password must be at least 6 characters.")
            else:
                with st.spinner("Creating account..."):
                    user = create_user(email, password)
                    if user:
                        token = generate_jwt(user['user_id'], user['email'])
                        
                        # Store token in session state
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        
                        # Save token to localStorage
                        set_token_in_storage(token)
                        
                        st.toast(f"✅ Account created! Welcome, {user['email']}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.toast("❌ User with this email already exists.")