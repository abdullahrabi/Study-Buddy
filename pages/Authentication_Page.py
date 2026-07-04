# pages/login.py - Login/Signup Page
import streamlit as st
import time
from auth import login_user, register_user, check_authentication

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

# Check if already authenticated
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

import os
load_css()

# ============================================
# HEADER (Original)
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

# ============================================
# LOGIN TAB (Using Flask API)
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
                    success, message = login_user(email, password)
                    if success:
                        st.toast(f"✅ Welcome back, {email}!")
                        time.sleep(0.5)
                        st.switch_page("pages/Main_Page.py")
                        st.stop()
                    else:
                        st.toast(f"❌ {message}")

# ============================================
# SIGNUP TAB (Using Flask API)
# ============================================
with tab2:
    with st.form("signup_form", clear_on_submit=True):
        email = st.text_input("Email Address", placeholder="you@example.com", key="signup_email")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            if not email or not password or not confirm_password:
                st.toast("❌ Please fill in all fields.")
            elif password != confirm_password:
                st.toast("❌ Passwords do not match.")
            elif len(password) < 8:
                st.toast("❌ Password must be at least 8 characters.")
            else:
                with st.spinner("Creating account..."):
                    success, message = register_user(email, password)
                    if success:
                        st.toast(f"✅ Account created! Welcome, {email}!")
                        time.sleep(0.5)
                        # Auto-login after registration
                        login_success, login_message = login_user(email, password)
                        if login_success:
                            st.switch_page("pages/Main_Page.py")
                            st.stop()
                        else:
                            st.toast("✅ Account created! Please login.")
                            st.rerun()
                    else:
                        st.toast(f"❌ {message}")