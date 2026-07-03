# pages/login.py - Login/Signup Page
import streamlit as st
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from auth import login_user, verify_user, init_session_state
import bcrypt
import jwt
import streamlit.components.v1 as components
import secrets

from auth import (
    init_session_state, 
    check_authentication, 
    verify_user, 
    create_user, 
    generate_jwt
)

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
# CHECK IF ALREADY AUTHENTICATED
# ============================================

if check_authentication():
    st.switch_page("pages/Main_Page.py")
    st.stop()

# ============================================
# LOAD CSS (Keep your existing CSS)
# ============================================

def load_css():
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
# HEADER
# ============================================

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

tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])

def save_token_to_localstorage(token):
    """Save token to localStorage via JavaScript"""
    js_code = f"""
    <script>
        localStorage.setItem('auth_token', '{token}');
        localStorage.setItem('token_timestamp', Date.now().toString());
        console.log('Token saved to localStorage');
    </script>
    """
    components.html(js_code, height=0)

# ============================================
# LOGIN TAB
# ============================================
with tab1:
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email Address", placeholder="you@example.com", key="login_email")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not email or not password:
                st.toast("❌ Please fill in all fields.")
            else:
                with st.spinner("Logging in..."):
                    user = verify_user(email, password)
                    if user:
                        token = generate_jwt(user['user_id'], user['email'])
                        
                        # Set session state (PRIMARY storage)
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        st.session_state.auth_checked = True
                        
                        # Save to localStorage (BACKUP)
                        save_token_to_localstorage(token)
                        
                        # Set token in query params (for refresh recovery)
                        st.query_params["token"] = token
                        login_user(user)  # This sets the cookie and session state
                        st.toast(f"✅ Welcome back, {user['email']}!")
                        time.sleep(0.5)
                        st.switch_page("pages/Main_Page.py")
                        st.stop()
                    else:
                        st.toast("❌ Invalid email or password.")

# ============================================
# SIGNUP TAB
# ============================================
with tab2:
    with st.form("signup_form", clear_on_submit=True):
        email = st.text_input("Email Address", placeholder="you@example.com", key="signup_email")
        password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
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
                        
                        # Set session state (PRIMARY storage)
                        st.session_state.token = token
                        st.session_state.user_id = user['user_id']
                        st.session_state.user_email = user['email']
                        st.session_state.logged_in = True
                        st.session_state.auth_checked = True
                        
                        # Save to localStorage (BACKUP)
                        save_token_to_localstorage(token)
                        
                        # Set token in query params (for refresh recovery)
                        st.query_params["token"] = token
                        
                        st.toast(f"✅ Account created! Welcome, {user['email']}!")
                        time.sleep(0.5)
                        st.switch_page("pages/Main_Page.py")
                        st.stop()
                    else:
                        st.toast("❌ User with this email already exists.")