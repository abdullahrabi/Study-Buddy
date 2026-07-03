# app.py - Login/Signup Page (With Time Gap for Async Issues)
import streamlit as st
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import bcrypt
import jwt
import streamlit.components.v1 as components
import secrets

# Import auth functions
from auth import init_session_state, check_authentication, verify_user, create_user, generate_jwt, verify_token

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    JWT_SECRET = secrets.token_urlsafe(32)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="StudyBuddy - Login",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# INITIALIZE SESSION STATE
# ============================================

init_session_state()

# ============================================
# CHECK AUTHENTICATION WITH TIME GAP
# ============================================

# Add small delay to handle async issues
time.sleep(0.1)

# Check if token is in URL
token_from_url = st.query_params.get("token", None)

if token_from_url and not st.session_state.auth_checked:
    try:
        # Add delay before validation
        time.sleep(0.1)
        
        decoded = jwt.decode(token_from_url, JWT_SECRET, algorithms=['HS256'])
        exp = decoded.get('exp')
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) > datetime.now(timezone.utc):
            st.session_state.token = token_from_url
            st.session_state.user_id = decoded.get('user_id')
            st.session_state.user_email = decoded.get('email')
            st.session_state.logged_in = True
            st.session_state.auth_checked = True
            st.query_params.clear()
            
            # Add delay before redirect
            time.sleep(0.2)
            st.switch_page("pages/main.py")
            st.stop()
        else:
            st.query_params.clear()
    except Exception as e:
        print(f"Token validation error: {e}")
        st.query_params.clear()

# If already logged in via session state
if st.session_state.logged_in and st.session_state.token:
    time.sleep(0.1)
    st.switch_page("pages/main.py")
    st.stop()

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
            .header-card {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 25px;
                text-align: center;
                margin-bottom: 20px;
            }
            .header-inner {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
            }
            .header-avatar {
                flex-shrink: 0;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #6366f1, #a855f7);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }
            .header-avatar img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                filter: brightness(0) invert(1);
            }
            .header-inner h1 {
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
            }
            .header-inner p {
                font-size: 13px;
                color: #94a3b8;
                margin: 5px 0 0 0;
                -webkit-text-fill-color: #94a3b8;
            }
            .stButton button {
                background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
            }
            .stTextInput input {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 12px !important;
                color: #e6eef9 !important;
                padding: 12px 16px !important;
            }
            .stTextInput input:focus {
                border-color: #6366f1 !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
            }
            .stToast {
                background: rgba(30, 35, 60, 0.95) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 12px !important;
                color: #e6eef9 !important;
            }
        </style>
        """, unsafe_allow_html=True)

load_css()

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
                        
                        # Set session state
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        st.session_state.auth_checked = True
                        
                        # Save to localStorage (just for backup)
                        components.html(f"""
                        <script>
                            localStorage.setItem('auth_token', '{token}');
                            localStorage.setItem('token_timestamp', Date.now().toString());
                            console.log('Token saved to localStorage');
                        </script>
                        """, height=0)
                        
                        # IMPORTANT: Set token in query params (this survives refresh)
                        st.query_params["token"] = token
                        
                        # Add delay before redirect to ensure everything is saved
                        time.sleep(0.5)
                        
                        st.toast(f"✅ Welcome back, {user['email']}!")
                        time.sleep(0.3)
                        st.switch_page("pages/main.py")
                        st.stop()
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
                        
                        # Set session state
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        st.session_state.auth_checked = True
                        
                        # Save to localStorage (just for backup)
                        components.html(f"""
                        <script>
                            localStorage.setItem('auth_token', '{token}');
                            localStorage.setItem('token_timestamp', Date.now().toString());
                            console.log('Token saved to localStorage');
                        </script>
                        """, height=0)
                        
                        # IMPORTANT: Set token in query params (this survives refresh)
                        st.query_params["token"] = token
                        
                        # Add delay before redirect to ensure everything is saved
                        time.sleep(1.5)
                        
                        st.toast(f"✅ Account created! Welcome, {user['email']}!")
                        time.sleep(0.3)
                        st.switch_page("pages/main.py")
                        st.stop()
                    else:
                        st.toast("❌ User with this email already exists.")