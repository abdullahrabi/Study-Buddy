import streamlit as st
import os
import tempfile
import base64
import time
from ai_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    generate_quiz_from_notes,
    evaluate_quiz_attempt,
    store_notes_and_progress,
    fetch_progress_from_pinecone,
    save_chat_history,
    fetch_chat_history,
    get_gemini_response,
    format_quiz_for_display,
)

# -----------------------------
# CSS STYLES - NOW AT THE TOP
# -----------------------------
st.markdown("""
<style>
    /* Sidebar background */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061022, #0b1523) !important;
        color: #e6eef9 !important;
    }

    /* Sidebar text and headings */
    section[data-testid="stSidebar"] * {
        color: #e6eef9 !important;
    }
    
    :root{
        --bg-1: #061022;
        --bg-2: #0b1523;
        --card: rgba(255,255,255,0.03);
        --glass: rgba(255,255,255,0.04);
        --accent1: #4fded8;
        --accent2: #8a7bff;
        --accent3: #ff6b9d;
        --accent4: #00d9ff;
        --accent5: #ffd166;
        --muted: #9aa3bf;
        --text: #e6eef9;
    }

/* ============================================
   CLEAN PROFESSIONAL TABS - COMPLETE VERSION
   ============================================ */

/* 1. SUBTLE TAB CONTAINER */
.stTabs {
    background: transparent !important;
    backdrop-filter: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 15px 0 25px 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* 2. CLEAN TAB LIST */
.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    gap: 2px !important;
    width: 100% !important;
    justify-content: flex-start !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* 3. MINIMAL INDIVIDUAL TAB */
.stTabs [data-baseweb="tab"] {
    flex: 1 !important;
    text-align: center !important;
    padding: 10px 10px !important;
    background: transparent !important;
    backdrop-filter: none !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    color: #9aa3bf !important;
    font-weight: 600 !important;
    font-size: 26px !important; 
    transition: all 0.25s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    min-height: 50px !important; /* Increased for big icons */
    cursor: pointer !important;
    line-height: 1.2 !important; /* Better text spacing */
}

/* NUCLEAR FONT SIZE OVERRIDE - Added to fix 14px issue */
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] > div > div,
.stTabs [data-baseweb="tab"] > div > div > div,
.stTabs [data-baseweb="tab"] span:not([role="img"]) {
    font-size: 26px !important !important; /* Double !important */
    font-weight: 600 !important;
    color: inherit !important;
}

/* Override any inline font-size from Streamlit */
.stTabs [data-baseweb="tab"][style*="font-size"],
.stTabs [data-baseweb="tab"] *[style*="font-size"] {
    font-size: 26px !important !important !important; /* Triple !important */
}

/* 4. SUBTLE HOVER */
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #d6e0f5 !important;
}

/* 5. ACTIVE TAB (Clean Blue Selection) */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(79, 222, 216, 0.15) !important;
    border: 1px solid rgba(79, 222, 216, 0.25) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 26px !important !important; /* Double !important */
}

/* Active tab text override */
.stTabs [data-baseweb="tab"][aria-selected="true"] *,
.stTabs [data-baseweb="tab"][aria-selected="true"] > div > div {
    font-size: 26px !important !important;
    color: white !important;
}

/* 6. REMOVE ALL GLOW/ANIMATION EFFECTS */
.stTabs [data-baseweb="tab"][aria-selected="true"]::before,
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    display: none !important;
}

/* 7. BIG ICONS - Increased size */
.stTabs [data-baseweb="tab"] img {
    width: 26px !important; /* Increased from 32px */
    height: 26px !important; /* Increased from 32px */
    min-width: 26px !important;
    min-height: 26px !important;
    object-fit: contain !important;
    margin: 0.2rem !important;
    transition: all 0.25s ease !important;
}

/* Override Streamlit's inline icon sizing */
.stTabs [data-baseweb="tab"] img[style*="width"],
.stTabs [data-baseweb="tab"] img[style*="height"] {
    width: 28px !important !important;
    height: 28px !important !important;
}

/* 8. ACTIVE ICON (Subtle Change) */
.stTabs [data-baseweb="tab"][aria-selected="true"] img {
    border-color: rgba(79, 222, 216, 0.3) !important;
    transform: scale(1.05) !important;
}

/* 9. EMOJI FALLBACK - Bigger to match icons */
.stTabs [data-baseweb="tab"] span[role="img"] {
    font-size: 32px !important; /* Increased */
    opacity: 0.9 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] span[role="img"] {
    opacity: 1 !important;
    transform: scale(1.05) !important;
}

/* 10. REMOVE DEFAULT HIGHLIGHT */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* 11. TAB CONTENT AREA - Fixed typo */
div[data-baseweb="tab-panel"] {
    padding: 40px 0 10px 0 !important; /* Fixed: was "40px 10 10px 0" */
}

/* 12. FOCUS STATES FOR ACCESSIBILITY */
.stTabs [data-baseweb="tab"]:focus-visible {
    outline: 2px solid rgba(79, 222, 216, 0.4) !important;
    outline-offset: 2px !important;
}

/* 13. EXTRA OVERRIDE FOR STUBBORN STREAMLIT STYLES */
div[data-testid="stTabs"] [data-baseweb="tab"],
div[data-testid="stTabs"] [data-baseweb="tab"] * {
    font-size: 18px !important;
}


html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, var(--bg-1), var(--bg-2)) !important;
        color: var(--text) !important;
        font-family: "Segoe UI", Inter, Roboto, -apple-system, Arial !important;
    }

    div[data-testid="stVerticalBlock"] {
        margin:0 !important;
        padding:0 !important;
    }

    .header-card {
        max-width:1100px;
        margin:0 auto 12px auto;
        padding:28px 28px 20px;
        border-radius:18px;
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
        box-shadow:0 8px 30px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.04);
        backdrop-filter:blur(8px)saturate(120%);
    }

    .header-inner {
        display:flex;
        gap:20px;
        align-items:center;
        justify-content:center;
    }

    .header-avatar img{
        width:90px;
        border-radius:8px;
    }

    .chat-box {
        border-radius:12px;
        padding:20px;
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
        border:1px solid rgba(255,255,255,0.03);
        box-shadow:0 8px 20px rgba(0,0,0,0.4);
        margin:10px auto;
        max-width:900px;
    }

    .quiz-box {
        border-radius:12px;
        padding:20px;
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
        border:1px solid rgba(255,255,255,0.03);
        box-shadow:0 8px 20px rgba(0,0,0,0.4);
        margin:10px auto;
        max-width:900px;
    }

    .avatar-img{
        width:200px;
        height:200px;
        display:block;
        margin:auto;
        border-radius:14px;
    }

    .msg-bubble {
        display:inline-block;
        padding:14px 16px;
        border-radius:14px;
        max-width:80%;
        font-size:15px;
        line-height:1.35;
        border:1px solid rgba(255,255,255,0.03);
        backdrop-filter:blur(6px)saturate(120%);
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
        box-shadow:0 8px 24px rgba(0,0,0,0.55);
    }

    .msg-user {
        margin-left:auto;
        background:linear-gradient(90deg,rgba(79,222,216,0.06),rgba(137,80,255,0.06));
        border:1px solid rgba(79,222,216,0.18);
    }

    .msg-bot {
        margin-right:auto;
        background:linear-gradient(90deg,rgba(137,80,255,0.03),rgba(79,222,216,0.02));
        border:1px solid rgba(137,80,255,0.08);
    }

    .msg-time {
        display:block;
        font-size:11px;
        color:var(--muted);
        margin-top:6px;
    }

    .input-wrap {
        display:flex;
        align-items:center;
        padding:12px;
        border-radius:12px;
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
        border:1px solid rgba(255,255,255,0.03);
        position:relative;
    }

    .chat-input {
        flex:1;
        border:none;
        background:transparent;
        padding:12px 16px 12px 16px;
        padding-right:60px;
        font-size:15px;
        color:var(--text);
    }

    .chat-input:focus { outline:none; }

    .send-btn {
        padding:10px 20px;
        border-radius:10px;
        border:none;
        cursor:pointer;
        background:linear-gradient(90deg,#2563eb,#3b82f6);
        color:white;
        font-weight:700;
        box-shadow:0 8px 24px rgba(37,99,235,0.25);
        position:absolute;
        right:12px;
        top:50%;
        transform:translateY(-50%);
        transition:0.2s ease;
    }

    .send-btn:hover {
        box-shadow:0 8px 24px rgba(37,99,235,0.35);
        transform:translateY(-50%)scale(1.02);
    }

    /* Interactive Option Cards - Direct Click Selection */
    .option-grid {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 25px 0;
    }
    
    .option-card-interactive {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 2px solid rgba(255,255,255,0.08);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        user-select: none;
        min-height: 80px;
    }
    
    .option-card-interactive:hover {
        transform: translateY(-3px);
        border-color: rgba(79,222,216,0.3);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }
    
    .option-card-interactive.selected {
        background: linear-gradient(135deg, rgba(138,123,255,0.2), rgba(79,222,216,0.15));
        border-color: rgba(138,123,255,0.6);
        box-shadow: 0 12px 35px rgba(138,123,255,0.25);
    }
    
    .option-card-interactive.selected::after {
        content: '✓';
        position: absolute;
        top: 12px;
        right: 12px;
        width: 28px;
        height: 28px;
        background: rgba(138,123,255,0.3);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 16px;
    }
    
    .option-card-interactive.correct {
        background: linear-gradient(135deg, rgba(79,222,216,0.15), rgba(0,217,255,0.08)) !important;
        border-color: rgba(79,222,216,0.5) !important;
        border-left: 4px solid #4fded8 !important;
    }
    
    .option-card-interactive.wrong {
        background: linear-gradient(135deg, rgba(255,107,157,0.15), rgba(255,142,107,0.08)) !important;
        border-color: rgba(255,107,157,0.5) !important;
        border-left: 4px solid #ff6b9d !important;
    }
    
    .option-card-interactive.partial {
        background: linear-gradient(135deg, rgba(255,193,7,0.2), rgba(255,165,0,0.15));
        border-color: rgba(255,193,7,0.6);
        border-left: 4px solid #ffd166 !important;
    }
    
    .option-label-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: rgba(138,123,255,0.15);
        border-radius: 10px;
        font-weight: bold;
        margin-right: 15px;
        color: #8a7bff;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .option-card-interactive.selected .option-label-circle {
        background: rgba(138,123,255,0.4);
        color: white;
        transform: scale(1.1);
    }
    
    .option-card-interactive.correct .option-label-circle {
        background: rgba(79,222,216,0.3) !important;
        color: white !important;
        border: 2px solid rgba(79,222,216,0.6) !important;
    }
    
    .option-card-interactive.wrong .option-label-circle {
        background: rgba(255,107,157,0.3) !important;
        color: white !important;
        border: 2px solid rgba(255,107,157,0.6) !important;
    }
    
    .option-card-interactive.partial .option-label-circle {
        background: rgba(255,209,102,0.3) !important;
        color: white !important;
        border: 2px solid rgba(255,193,7,0.6) !important;
    }
    
    .option-card-interactive .option-label-circle {
        transition: all 0.3s ease !important;
        font-weight: bold !important;
    }
    
    .option-content {
        flex: 1;
    }
    
    .option-text {
        font-size: 15px;
        line-height: 1.4;
        font-weight: 500;
    }
    
    .option-hint {
        font-size: 11px;
        color: var(--muted);
        margin-top: 4px;
        font-style: italic;
    }
    
    /* Single vs Multiple mode indicators */
    .mode-indicator {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
    }
    
    .single-mode {
        background: rgba(79,222,216,0.2);
        color: #4fded8;
    }
    
    .multi-mode {
        background: rgba(138,123,255,0.2);
        color: #8a7bff;
    }
    
    /* Difficulty-specific styles */
    .difficulty-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .difficulty-easy {
        background: rgba(79,222,216,0.2);
        color: #4fded8;
    }
    
    .difficulty-medium {
        background: rgba(138,123,255,0.2);
        color: #8a7bff;
    }
    
    .difficulty-hard {
        background: rgba(255,107,157,0.2);
        color: #ff6b9d;
    }
    
    .difficulty-difficult {
        background: rgba(255,65,108,0.2);
        color: #ff416c;
    }
    
    .difficult-warning-box {
        background: linear-gradient(135deg, rgba(255,107,157,0.1), rgba(255,65,108,0.05));
        border-left: 4px solid #ff6b9d;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .single-mode-box {
        background: linear-gradient(135deg, rgba(79,222,216,0.1), rgba(0,217,255,0.05));
        border-left: 4px solid #4fded8;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .multi-mode-box {
        background: linear-gradient(135deg, rgba(138,123,255,0.1), rgba(179,102,255,0.05));
        border-left: 4px solid #8a7bff;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }

    /* Enhanced Progress Bars */
    .progress-container {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar-gradient {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, 
            #4fded8 0%, 
            #8a7bff 50%, 
            #ff6b9d 100%);
        position: relative;
        transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 20px rgba(79, 222, 216, 0.3);
    }
    
    .progress-bar-gradient::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.2), 
            transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 12px;
        color: var(--muted);
    }
    
    .progress-percentage {
        font-weight: bold;
        color: var(--accent2);
    }
    
    /* Modern Submit Button */
    .modern-submit-btn {
        width: 100%;
        padding: 18px;
        border-radius: 14px;
        background: linear-gradient(90deg, #8a7bff, #4fded8);
        color: white;
        font-weight: bold;
        font-size: 17px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(138, 123, 255, 0.3);
        letter-spacing: 0.5px;
    }
    
    .modern-submit-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(138, 123, 255, 0.4);
        background: linear-gradient(90deg, #7a6bff, #3fced8);
    }
    
    .modern-submit-btn:disabled {
        background: linear-gradient(90deg, #9aa3bf, #8a93af);
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
        opacity: 0.6;
    }
    
    /* Modern Navigation Buttons */
    .nav-btn-container {
        display: flex;
        gap: 12px;
        margin-top: 20px;
    }
    
    .nav-btn {
        flex: 1;
        padding: 14px;
        border-radius: 12px;
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.1);
        color: var(--text);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
    }
    
    .nav-btn:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(79,222,216,0.4);
        transform: translateY(-2px);
    }
    
    .nav-btn-primary {
        background: linear-gradient(90deg, #8a7bff, #4fded8);
        border: none;
        color: white;
        font-weight: bold;
    }
    
    .nav-btn-primary:hover {
        background: linear-gradient(90deg, #7a6bff, #3fced8);
        box-shadow: 0 10px 25px rgba(138, 123, 255, 0.3);
    }

    /* Quiz progress indicator */
    .quiz-progress-container {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .quiz-progress-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    
    .quiz-progress-stats {
        display: flex;
        gap: 15px;
        font-size: 14px;
    }
    
    .quiz-progress-stat {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .quiz-progress-dots {
        display: flex;
        gap: 8px;
        margin-top: 15px;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        transition: all 0.3s;
    }
    
    /* Progress dot colors - Beautiful unified scheme */
    .progress-dot.active {
        background: linear-gradient(135deg, #8a7bff, #b366ff) !important;
        box-shadow: 0 0 10px rgba(138, 123, 255, 0.8) !important;
    }
    
    .progress-dot.correct {
        background: linear-gradient(135deg, #4fded8, #00d9ff) !important;
        box-shadow: 0 0 10px rgba(79, 222, 216, 0.8) !important;
    }
    
    .progress-dot.incorrect {
        background: linear-gradient(135deg, #ff6b9d, #ff416c) !important;
        box-shadow: 0 0 10px rgba(255, 107, 157, 0.8) !important;
    }
    
    .progress-dot.partial {
        background: linear-gradient(135deg, #ffd166, #ffb74d) !important;
        box-shadow: 0 0 10px rgba(255, 209, 102, 0.8) !important;
    }
    
    /* Selection counter */
    .selection-counter {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(138,123,255,0.2);
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        color: #8a7bff;
        margin-left: 10px;
    }

    /* Circle progress bar container */
    .circle-progress-container {
        text-align: center;
        margin: 20px 0;
    }

    /* Clickable option card */
    .clickable-option {
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .clickable-option:hover {
        transform: translateY(-2px);
    }
    
    /* Make buttons invisible but clickable */
    .invisible-button {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        z-index: 10;
        padding: 0;
        margin: 0;
        border: none;
        background: transparent;
    }
    
    .option-card-container {
        position: relative;
    }

    /* Progress bar with status-based gradient */
    .progress-bar-status {
        height: 100%;
        border-radius: 10px;
        position: relative;
        transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 20px rgba(79, 222, 216, 0.3);
    }
    
    .progress-bar-status::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.2), 
            transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Multi-segment progress bar */
    .multi-segment-progress {
        width: 100%;
        height: 16px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
        display: flex;
    }
    
    .progress-segment {
        height: 100%;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-segment::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.3), 
            transparent);
        animation: shimmer 3s infinite;
    }
    
    .segment-correct {
        background: linear-gradient(135deg, #4fded8, #00d9ff);
        box-shadow: 0 0 15px rgba(79, 222, 216, 0.4);
    }
    
    .segment-partial {
        background: linear-gradient(135deg, #ffd166, #ffb74d);
        box-shadow: 0 0 15px rgba(255, 209, 102, 0.4);
    }
    
    .segment-incorrect {
        background: linear-gradient(135deg, #ff6b9d, #ff416c);
        box-shadow: 0 0 15px rgba(255, 107, 157, 0.4);
    }
    
    .segment-active {
        background: linear-gradient(135deg, #8a7bff, #b366ff);
        box-shadow: 0 0 15px rgba(138, 123, 255, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .segment-pending {
        background: rgba(255,255,255,0.1);
    }
    
    /* Footer tab navigation styling */
    div[data-testid="stHorizontalBlock"] {
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    
    /* Make tab buttons more visible */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #8a7bff, #4fded8) !important;
        font-weight: bold !important;
        color: white !important;
    }
    
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #9aa3bf !important;
    }
    
    /* Progress tab metric cards */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.05);
        height: 100%;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
        background: linear-gradient(90deg, #4fded8, #8a7bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 14px;
        color: #9aa3bf;
        margin-bottom: 10px;
    }
    
    .metric-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    /* Legend container */
    .legend-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        margin: 20px 0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
    }
    
    .legend-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }
    
    .legend-text {
        font-size: 14px;
        color: #e6eef9;
    }
    
    .legend-description {
        font-size: 12px;
        color: #9aa3bf;
        margin-left: 26px;
    }

    
/* FIXED: Custom GIF Loader Styles - Reduced gap between loader and text */
.custom-loader-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 2px !important;  /* REDUCED from 5px to 2px */
    padding: 10px;
    margin: 10px 0;
    text-align: center;
    flex-wrap: nowrap;
}

.custom-loader {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50%;
    margin: 0 !important;
}

.custom-loader-text {
    color: var(--muted);
    font-size: 22px !important;  
    font-weight: 600 !important;
    margin-left: 0 !important;
    padding-left: 2px !important;  /* REDUCED from 5px to 2px */
    display: inline-flex !important;
    align-items: center !important;
}

/* Hide default spinner containers */
[data-testid="stSpinner"] {
    display: none !important;
}

.stSpinner {
    display: none !important;
}

/* NEW: Button Gradient Styles for Notes & Quiz and Progress Sections */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #8a7bff, #4fded8) !important;
    font-weight: bold !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(138, 123, 255, 0.3) !important;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(90deg, rgba(138, 123, 255, 0.1), rgba(79, 222, 216, 0.1)) !important;
    border: 1px solid rgba(138, 123, 255, 0.3) !important;
    color: #8a7bff !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #7a6bff, #3fced8) !important;
    box-shadow: 0 12px 25px rgba(138, 123, 255, 0.4) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: linear-gradient(90deg, rgba(138, 123, 255, 0.2), rgba(79, 222, 216, 0.2)) !important;
    border-color: rgba(138, 123, 255, 0.5) !important;
}

/* Chat section specific button styles - keep original */
[data-testid="stForm"] [data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border: none !important;
}

[data-testid="stForm"] [data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
}

/* Direct click option cards - no button overlay needed */
.option-card-clickable {
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.option-card-clickable:hover {
    border-color: rgba(79,222,216,0.4) !important;
    transform: translateY(-2px) !important;
}

/* Simplified direct click option */
.option-direct-click {
    position: relative;
    width: 100%;
    height: 100%;
}

/* Ensure proper spacing for loader in chat */
[data-testid="stVerticalBlock"] > div > .element-container:has(.custom-loader-container) {
    margin: 0 !important;
    padding: 0 !important;
}

/* NEW: Circle progress bar that behaves like progress line - FIXED to use consistent gradient */
.circle-progress-line {
    position: relative;
    width: 100px;
    height: 100px;
    margin: 0 auto;
}

.circle-progress-line svg {
    transform: rotate(-90deg);
}

.circle-progress-line .progress-background {
    fill: none;
    stroke: rgba(255,255,255,0.1);
    stroke-width: 8;
}

.circle-progress-line .progress-fill {
    fill: none;
    stroke-linecap: round;
    stroke-width: 8;
    transition: stroke-dashoffset 0.5s ease;
}

.circle-progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    width: 100%;
}

.circle-progress-value {
    font-size: 20px;
    font-weight: bold;
    color: #e6eef9;
    line-height: 1;
}

.circle-progress-label {
    font-size: 12px;
    color: #9aa3bf;
    margin-top: 2px;
}
.report-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 50px;
    width: 50px;
    margin-right: 0px;
    border-radius: 8px;
    object-fit: contain;
    position: relative;
    background: transparent;
    padding: 2px;
}

.report-icon::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    z-index: -1;
}
/* ============================================
   BIG CALENDAR ICON FIX - FINAL SOLUTION
   ============================================ */

/* Target ALL images with calendar-icon-class in expanders */
div[data-testid="stExpander"] img[alt="calendar-icon-class"] {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    object-fit: contain !important;
    transition: all 0.3s ease !important;
}



/* Target the specific structure of expander headers */
div[data-testid="stExpander"] > div:first-child > div:first-child img[alt="calendar-icon-class"],
div[data-testid="stExpander"]Details img[alt="calendar-icon-class"],
.stExpander img[alt="calendar-icon-class"] {
    width: 40px !important;
    height: 40px !important;
}

/* Nuclear option - target any image with calendar-icon-class anywhere */
img[alt="calendar-icon-class"] {
    width: 40px !important;
    height: 40px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Base64 Image/GIF Loader
# -----------------------------
def get_base64_file(file_path):
    """Read any file and return base64 encoded data"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return ""

# Then update your image loading:
Avatar_Icon = get_base64_file("Bot_Avatar.png") if os.path.exists("Bot_Avatar.png") else ""
tab_chat_icon = get_base64_file("tab_chat_icon.png") if os.path.exists("tab_chat_icon.png") else ""
Spinner_Loader = get_base64_file("Spinner_Loader.gif") if os.path.exists("Spinner_Loader.gif") else ""
notes_icon = get_base64_file("notes.png") if os.path.exists("notes.png") else ""
progress_icon = get_base64_file("progress.png") if os.path.exists("progress.png") else ""
report_icon = get_base64_file("3d-report.png") if os.path.exists("3d-report.png") else ""
Calendar_Icon = get_base64_file("Calendar.png") if os.path.exists("Calendar.png") else ""



# Also update the show_custom_loader function to properly use base64 for GIFs:
def show_custom_loader(text="Processing..."):
    """Display custom GIF loader using base64 with minimal gap"""
    if Spinner_Loader:
        # FIXED: Reduced gap between loader and text
        loader_html = f"""
        <div class="custom-loader-container">
            <img src="data:image/gif;base64,{Spinner_Loader}" class="custom-loader" alt="Loading...">
            <div class="custom-loader-text">{text}</div>
        </div>
        """
    else:
        # Fallback to CSS spinner
        loader_html = f"""
        <div class="custom-loader-container">
            <div style="width: 40px; height: 40px; margin: 0; position: relative;">
                <div style="position: absolute; width: 100%; height: 100%; border: 4px solid rgba(138, 123, 255, 0.2); border-radius: 50%;"></div>
                <div style="position: absolute; width: 100%; height: 100%; border: 4px solid transparent; border-top: 4px solid #8a7bff; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
            <div class="custom-loader-text">{text}</div>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
    return loader_html

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Study Assistant — StudyBuddy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State Setup
# -----------------------------
defaults = {
    "user_id": "default_user",
    "attempt": 1,
    "notes_text": "",
    "quiz": None,
    "current_q": 0,
    "answers": {},
    "completed": False,
    "chat_sessions": [],
    "current_session_id": None,
    "quiz_source": "notes",
    "difficulty": "medium",
    "num_questions": 5,
    "custom_topic": "",
    "chat_input": "",
    "last_sent_message": "",
    "uploaded_file": None,
    "selected_options": {},   # Track selected options for current question
    "question_answered": {},  # Track which questions have been answered
    "show_feedback": {},      # Track if feedback should be shown for each question
    "answer_submitted": {},   # Track if answer has been submitted for each question
    "question_status": {},    # Track status of each question (correct/incorrect/partial)
    "last_clicked_option": None,  # Track last clicked option for interactivity
    "need_rerun": False,      # Flag to trigger rerun
    "active_tab": "Notes & Quiz",  # Track active tab for navigation
    "ai_responding": False,   # Track if AI is responding in chat
    "streaming_message": None,  # Track the streaming message
    "streaming_finished": False,  # Track if streaming is finished
    "current_accuracy": 0,    # Track current accuracy for progress bar colors
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Chat Helpers
# -----------------------------
def generate_topic_name(first_message):
    words = first_message.split()[:4]
    topic = " ".join(words)
    if len(first_message) > 30:
        topic += "..."
    return topic

def create_new_chat_session(first_message):
    session_id = f"session_{int(time.time())}"
    topic_name = generate_topic_name(first_message)

    new_session = {
        "session_id": session_id,
        "topic": topic_name,
        "first_message": first_message,
        "created_at": time.time(),
        "messages": [
            {"role": "user", "message": first_message, "timestamp": time.time()}
        ],
    }

    st.session_state.chat_sessions.append(new_session)
    st.session_state.current_session_id = session_id
    return new_session

def get_current_session_messages():
    if not st.session_state.current_session_id:
        return []
    for session in st.session_state.chat_sessions:
        if session["session_id"] == st.session_state.current_session_id:
            return session["messages"]
    return []

def add_message_to_current_session(role, message):
    if not st.session_state.current_session_id:
        return
    for session in st.session_state.chat_sessions:
        if session["session_id"] == st.session_state.current_session_id:
            session["messages"].append(
                {"role": role, "message": message, "timestamp": time.time()}
            )
            break

# -----------------------------
# Interactive Quiz Helpers
# -----------------------------
def extract_text(file_path):
    """Extract text from uploaded file based on extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def handle_option_click(question_idx, option_key):
    """Handle option click - direct selection without extra buttons"""
    if question_idx not in st.session_state.selected_options:
        st.session_state.selected_options[question_idx] = set()
    
    current_selection = st.session_state.selected_options[question_idx]
    
    # Get answer type from quiz data
    answer_type = "single"
    if st.session_state.quiz and question_idx < len(st.session_state.quiz.get("quiz", [])):
        answer_type = st.session_state.quiz["quiz"][question_idx].get("answer_type", "single")
    
    is_difficult = st.session_state.difficulty == "Difficult"
    
    if answer_type == "single" and not is_difficult:
        # For single answer (not Difficult mode), replace selection
        st.session_state.selected_options[question_idx] = {option_key}
    else:
        # For multiple answer OR Difficult mode, toggle selection
        if option_key in current_selection:
            current_selection.remove(option_key)
        else:
            current_selection.add(option_key)
    
    # Update last clicked for visual feedback
    st.session_state.last_clicked_option = (question_idx, option_key)
    st.session_state.need_rerun = True

def submit_answer(question_idx, answer_type, show_feedback=True):
    """Submit the current answer"""
    if question_idx in st.session_state.selected_options:
        selected = st.session_state.selected_options[question_idx]
        if selected:
            if answer_type == "single":
                # Store as single letter
                st.session_state.answers[question_idx] = list(selected)[0].upper()
            else:
                # Store as comma-separated string, sorted alphabetically
                st.session_state.answers[question_idx] = ",".join(sorted([opt.upper() for opt in list(selected)]))
            
            st.session_state.question_answered[question_idx] = True
            st.session_state.answer_submitted[question_idx] = True
            
            # Always show feedback for all modes now
            st.session_state.show_feedback[question_idx] = True
            
            # Clear selection for next question
            if question_idx in st.session_state.selected_options:
                del st.session_state.selected_options[question_idx]
            st.session_state.last_clicked_option = None
            st.session_state.need_rerun = True
        else:
            st.session_state.need_rerun = True
    else:
        st.session_state.need_rerun = True

# -----------------------------
# NEW: Multi-Segment Progress Bar Functions
# -----------------------------
def create_multi_segment_progress_bar(total_questions):
    """Create a beautiful multi-segment progress bar showing each question's status"""
    if not total_questions:
        return ""
    
    segments_html = ""
    segment_width = 100 / total_questions
    
    for i in range(total_questions):
        status = st.session_state.question_status.get(i, "")
        
        if i < st.session_state.current_q:
            # Past questions
            if status == "correct":
                segment_class = "segment-correct"
            elif status == "partial":
                segment_class = "segment-partial"
            elif status == "incorrect":
                segment_class = "segment-incorrect"
            else:
                segment_class = "segment-pending"
        elif i == st.session_state.current_q:
            # Current question
            if status:  # Already answered current question
                if status == "correct":
                    segment_class = "segment-correct"
                elif status == "partial":
                    segment_class = "segment-partial"
                elif status == "incorrect":
                    segment_class = "segment-incorrect"
                else:
                    segment_class = "segment-pending"
            else:
                segment_class = "segment-active"  # Current active question
        else:
            # Future questions
            segment_class = "segment-pending"
        
        segments_html += f'<div class="progress-segment {segment_class}" style="width: {segment_width}%;"></div>'
    
    # Calculate percentages
    completed_count = len([i for i in range(total_questions) if i in st.session_state.question_answered])
    progress_percentage = (completed_count / total_questions * 100) if total_questions > 0 else 0
    
    return f"""
    <div style="margin: 20px 0;">
        <div class="multi-segment-progress">
            {segments_html}
        </div>
        <div class="progress-label">
            <span>Progress: {completed_count}/{total_questions} questions</span>
            <span class="progress-percentage">{progress_percentage:.1f}%</span>
        </div>
    </div>
    """

def get_progress_colors(accuracy_percentage):
    """Get unified color scheme based on accuracy percentage"""
    if accuracy_percentage >= 80:
        return {
            "primary": "#4fded8",  # Green/Teal
            "secondary": "#00d9ff",  # Bright Teal
            "gradient": "linear-gradient(135deg, #4fded8 0%, #00d9ff 100%)",
            "status": "excellent",
            "icon": "🎯"
        }
    elif accuracy_percentage >= 60:
        return {
            "primary": "#8a7bff",  # Purple
            "secondary": "#b366ff",  # Bright Purple
            "gradient": "linear-gradient(135deg, #8a7bff 0%, #b366ff 100%)",
            "status": "good",
            "icon": "👍"
        }
    elif accuracy_percentage >= 40:
        return {
            "primary": "#ff6b9d",  # Pink
            "secondary": "#ff8e6b",  # Orange
            "gradient": "linear-gradient(135deg, #ff6b9d 0%, #ff8e6b 100%)",
            "status": "fair",
            "icon": "💪"
        }
    else:
        return {
            "primary": "#ff416c",  # Red
            "secondary": "#ff4b2b",  # Dark Red
            "gradient": "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)",
            "status": "needs_improvement",
            "icon": "🔥"
        }

def get_current_accuracy():
    """Calculate current accuracy based on answered questions"""
    answered_questions = [i for i in st.session_state.question_answered.keys()]
    
    if not answered_questions:
        return 0
    
    correct_count = sum(1 for i in answered_questions if st.session_state.question_status.get(i) == "correct")
    partial_count = sum(1 for i in answered_questions if st.session_state.question_status.get(i) == "partial")
    
    # Weight partial as 0.5 correct
    weighted_correct = correct_count + (partial_count * 0.5)
    total_weighted = len(answered_questions)
    
    if total_weighted > 0:
        accuracy = (weighted_correct / total_weighted) * 100
    else:
        accuracy = 0
    
    return accuracy

def create_gradient_progress_bar(percentage, difficulty="medium", scenario="default"):
    """Create a beautiful gradient progress bar based on scenario"""
    # Beautiful gradient colors for different scenarios
    gradients = {
        "default": "linear-gradient(135deg, #4fded8 0%, #8a7bff 50%, #ff6b9d 100%)",
        "excellent": "linear-gradient(135deg, #4fded8 0%, #00d9ff 100%)",
        "good": "linear-gradient(135deg, #8a7bff 0%, #b366ff 100%)",
        "fair": "linear-gradient(135deg, #ff6b9d 0%, #ff8e6b 100%)",
        "needs_improvement": "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)",
        "easy": "linear-gradient(135deg, #4fded8 0%, #00d9ff 100%)",
        "medium": "linear-gradient(135deg, #8a7bff 0%, #b366ff 100%)",
        "hard": "linear-gradient(135deg, #ff6b9d 0%, #ff8e6b 100%)",
        "difficult": "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)",
        "success": "linear-gradient(135deg, #4fded8 0%, #00d9ff 100%)",
        "warning": "linear-gradient(135deg, #ffd166 0%, #ffb74d 100%)",
        "error": "linear-gradient(135deg, #ff6b9d 0%, #ff416c 100%)",
        "quiz_active": "linear-gradient(135deg, #4fded8 0%, #8a7bff 50%, #ff6b9d 100%)",
        "quiz_completed": "linear-gradient(135deg, #4fded8 0%, #00d9ff 100%)"
    }
    
    # Determine gradient based on scenario and difficulty
    if scenario in gradients:
        gradient = gradients[scenario]
    elif difficulty.lower() in gradients:
        gradient = gradients[difficulty.lower()]
    else:
        gradient = gradients["default"]
    
    return f"""
    <div class="progress-container">
        <div class="progress-bar-status" style="width: {percentage}%; background: {gradient};">
        </div>
    </div>
    <div class="progress-label">
        <span>Progress</span>
        <span class="progress-percentage">{percentage:.1f}%</span>
    </div>
    """

def create_quiz_progress_indicator(current_idx, total_questions, difficulty="medium"):
    """Create a visual quiz progress indicator with unified colors"""
    # Calculate current accuracy
    accuracy = get_current_accuracy()
    
    # Get unified color scheme based on accuracy
    colors = get_progress_colors(accuracy)
    
    # Generate progress dots with appropriate colors
    dots_html = ""
    for i in range(total_questions):
        status_class = ""
        if i < current_idx:
            # For previous questions, show their status
            status = st.session_state.question_status.get(i, "")
            if status == "correct":
                status_class = "correct"
            elif status == "partial":
                status_class = "partial"
            elif status == "incorrect":
                status_class = "incorrect"
            else:
                status_class = ""  # Not answered yet (shouldn't happen for i < current_idx)
        elif i == current_idx:
            status_class = "active"  # Current question
        
        dots_html += f'<div class="progress-dot {status_class}"></div>'
    
    # NEW: Create multi-segment progress bar
    segment_bar = create_multi_segment_progress_bar(total_questions)
    
    return {
        "header": f"""
        <div class="quiz-progress-header">
            <div>Question {current_idx + 1} of {total_questions}</div>
            <div class="quiz-progress-stats">
                <div class="quiz-progress-stat">
                    <div class="progress-dot correct"></div>
                    <span>Correct</span>
                </div>
                <div class="quiz-progress-stat">
                    <div class="progress-dot incorrect"></div>
                    <span>Incorrect</span>
                </div>
                <div class="quiz-progress-stat">
                    <div class="progress-dot partial"></div>
                    <span>Partial</span>
                </div>
                <div class="quiz-progress-stat">
                    <div class="progress-dot active"></div>
                    <span>Active</span>
                </div>
            </div>
        </div>
        """,
        "segment_bar": segment_bar,  # NEW: Multi-segment bar
        "dots": f'<div class="quiz-progress-dots">{dots_html}</div>',
        "colors": colors,
        "accuracy": accuracy
    }

# FIXED: Circle progress bar that behaves like progress line - consistent gradient throughout quiz
def create_circle_progress_bar(completed_count, total_questions, current_question_idx):
    """Create a circle progress bar that behaves like progress bar line - consistent gradient"""
    # Calculate progress percentage
    progress_percentage = (completed_count / total_questions * 100) if total_questions > 0 else 0
    
    # Use the SAME gradient as the progress bar line throughout the quiz
    # This ensures consistency and prevents it from turning red
    circle_gradient = "url(#circleGradient)"
    
    # SVG circle properties
    radius = 45
    circumference = 2 * 3.14159 * radius
    # Calculate the stroke-dashoffset based on progress
    # 0% progress = full circle hidden, 100% progress = full circle visible
    dash_offset = circumference - (progress_percentage / 100 * circumference)
    
    return f"""
    <div class="circle-progress-line">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <defs>
                <linearGradient id="circleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#4fded8" />
                    <stop offset="50%" stop-color="#8a7bff" />
                    <stop offset="100%" stop-color="#ff6b9d" />
                </linearGradient>
            </defs>
            <circle class="progress-background" cx="50" cy="50" r="{radius}" />
            <circle class="progress-fill" cx="50" cy="50" r="{radius}" 
                    stroke="{circle_gradient}"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{dash_offset}"
                    data-percentage="{progress_percentage}" />
        </svg>
        <div class="circle-progress-text">
            <div class="circle-progress-value">{completed_count}/{total_questions}</div>
            <div class="circle-progress-label">Question {current_question_idx + 1}</div>
        </div>
    </div>
    """

def get_difficulty_instructions(difficulty, answer_type=None):
    """Get instructions based on difficulty level"""
    if difficulty == "Difficult":
        return """
        <div class='difficult-warning-box'>
            <strong>🎯 DIFFICULT MODE:</strong><br>
            • Questions may have <strong>SINGLE OR MULTIPLE</strong> correct answers<br>
            • You won't know which type until submission<br>
            • <strong>Click on options</strong> to select/deselect them<br>
            • Full feedback shown after submission
        </div>
        """
    elif difficulty in ["Easy", "Medium", "Hard"]:
        if answer_type == "multiple":
            return """
            <div class='multi-mode-box'>
                <strong>🔢 MULTIPLE CORRECT ANSWERS:</strong><br>
                • This question has <strong>MULTIPLE</strong> correct answers<br>
                • <strong>Click on options</strong> to select/deselect them<br>
                • Select all that you think are correct
            </div>
            """
        else:
            return """
            <div class='single-mode-box'>
                <strong>📝 SINGLE CORRECT ANSWER:</strong><br>
                • This question has a <strong>SINGLE</strong> correct answer<br>
                • <strong>Click on one option</strong> to select it
            </div>
            """
    return ""

# UPDATED FUNCTION: Direct clickable options using st.button for all modes
def display_clickable_options(question_idx, options, current_selection, answer_type, difficulty):
    """Display option cards that can be directly clicked - Simplified version"""
    
    st.markdown('<div class="option-grid">', unsafe_allow_html=True)
    
    for option_key, option_text in options.items():
        is_selected = option_key in current_selection
        selected_class = "selected" if is_selected else ""
        
        # Create a container with a button that covers the entire card
        container = st.container()
        with container:
            # Display the card
            card_html = f"""
            <div class="option-card-interactive {selected_class} option-card-clickable" id="option_{question_idx}_{option_key}">
                <div style="display: flex; align-items: center;">
                    <div class="option-label-circle">{option_key}</div>
                    <div class="option-content">
                        <div class="option-text">{option_text}</div>
                        <div class="option-hint">Click to {'deselect' if is_selected else 'select'}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Create a transparent button that covers the entire card
            button_key = f"option_btn_{question_idx}_{option_key}"
            clicked = st.button(
                f"Select {option_key}",
                key=button_key,
                help=f"Select option {option_key}",
                type="secondary",
                # Make button invisible
                use_container_width=True
            )
            
            if clicked:
                # Handle the click
                if question_idx not in st.session_state.selected_options:
                    st.session_state.selected_options[question_idx] = set()
                
                current_selection = st.session_state.selected_options[question_idx]
                
                if difficulty == "Difficult":
                    # Difficult mode - always allow multiple selection
                    if option_key in current_selection:
                        current_selection.remove(option_key)
                    else:
                        current_selection.add(option_key)
                elif answer_type == "single":
                    # Single answer - replace selection
                    st.session_state.selected_options[question_idx] = {option_key}
                else:
                    # Multiple answer - toggle selection
                    if option_key in current_selection:
                        current_selection.remove(option_key)
                    else:
                        current_selection.add(option_key)
                
                st.session_state.last_clicked_option = (question_idx, option_key)
                st.session_state.need_rerun = True
        
        # CSS to make the button invisible but cover the card
        st.markdown(f"""
        <style>
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="element-container"] > div[data-testid="stButton"] > button[data-testid="stButton"][data-testkey="{button_key}"]) {{
            position: relative !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        button[data-testid="stButton"][data-testkey="{button_key}"] {{
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 80px !important;
            opacity: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            background: transparent !important;
            cursor: pointer !important;
            z-index: 20 !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """<div class="header-card"><div class="header-inner">
       <div class="header-avatar">
         <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
       </div>
       <div><h1>Meet StudyBuddy</h1>
         <p>Your Personal AI Learning Assistant with Advanced Quiz Features</p></div>
    </div></div>""",
    unsafe_allow_html=True,
)

# -----------------------------
# SIDEBAR: Chat History
# -----------------------------
with st.sidebar:
   
    if Avatar_Icon:
        st.markdown(
            f'<img src="data:image/png;base64,{Avatar_Icon}" class="avatar-img">',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="avatar-img">',
            unsafe_allow_html=True,
        )
    st.markdown("###  Welcome to **Study Buddy 👋**")
    st.markdown("### 💬 Chat Sessions")

    if st.button("🆕 Start New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.rerun()

    st.markdown("---")

    if st.session_state.chat_sessions:
        st.markdown('<div class="sidebar-chat-history">', unsafe_allow_html=True)
        for i, session in enumerate(reversed(st.session_state.chat_sessions[-8:])):
            active = session["session_id"] == st.session_state.current_session_id
            active_css = (
                "border:1px solid rgba(79,222,216,0.3);"
                "background:linear-gradient(90deg,rgba(79,222,216,0.08),rgba(137,80,255,0.06));"
                if active else ""
            )

            st.markdown(
                f'<div class="sidebar-chat-item" style="{active_css}">'
                f'<strong>💬 {session["topic"]}</strong><br>'
                f'<small>{time.strftime("%I:%M %p", time.localtime(session["created_at"]))}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
            st.session_state.chat_sessions = []
            st.session_state.current_session_id = None
            try:
                save_chat_history(st.session_state.user_id, [])
            except:
                pass
            st.rerun()
    else:
        st.info("No chat sessions yet!")

# -----------------------------
# MAIN TABS WITH FIXED ICONS
# -----------------------------
if tab_chat_icon:
    chat_icon = f"![Icon](data:image/png;base64,{tab_chat_icon}) "
    unsafe_allow_html=True
else:
    chat_icon = "💬"  # Use emoji instead of base64
if notes_icon:
    notes_icon = f"![Icon](data:image/png;base64,{notes_icon}) "
    unsafe_allow_html=True
else:
    notes_icon = "📝"
if progress_icon:
    progress_icon = f"![Icon](data:image/png;base64,{progress_icon}) "
    unsafe_allow_html=True
else:
    progress_icon = "📊"

# Create tabs with simple emojis
tab1, tab2, tab3 = st.tabs([
    f"{chat_icon} Chat",
    f"{notes_icon} Notes & Quiz", 
    f"{progress_icon} Progress"
])

# ---------- TAB 1: CHAT ----------
with tab1:
    # Update active tab
    st.session_state.active_tab = "Chat"
    
    current_msgs = get_current_session_messages()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for item in current_msgs:
            role = item["role"]
            txt = item["message"]
            ts = item["timestamp"]
            tstr = time.strftime("%I:%M %p", time.localtime(ts))

            if role == "user":
                st.markdown(
                    f'<div style="text-align:right; margin:10px 0;"><div class="msg-bubble msg-user">'
                    f"👤 {txt}<span class='msg-time'>{tstr}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                # Regular message
                st.markdown(
                    f'<div style="text-align:left; margin:10px 0;"><div class="msg-bubble msg-bot">'
                    f"🤖 {txt}<span class='msg-time'>{tstr}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        
        # Show custom loader in chat when AI is responding
        if st.session_state.get("ai_responding", False):
            st.markdown(show_custom_loader("Thinking..."), unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_msg = st.text_input(
                "Type your message",
                key="chat_input",
                placeholder="Ask me anything about your studies...",
                label_visibility="collapsed"
            )
        with col2:
            send_btn = st.form_submit_button("Send", use_container_width=True)
        
        if send_btn and user_msg.strip():
            if not st.session_state.current_session_id:
                create_new_chat_session(user_msg.strip())
            else:
                add_message_to_current_session("user", user_msg.strip())
            
            # Set AI responding flag
            st.session_state.ai_responding = True
            st.session_state.last_user_message = user_msg.strip()
            st.session_state.last_chat_messages = current_msgs
            
            # Trigger response
            st.rerun()

# Handle AI response after rerun
if st.session_state.get("ai_responding", False) and st.session_state.get("last_user_message"):
    try:
        # Get the streaming response
        response_stream = get_gemini_response(
            st.session_state.last_user_message, 
            st.session_state.last_chat_messages, 
            st.session_state.user_id
        )
        
        # Collect the full response
        full_response = ""
        for chunk in response_stream:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
            elif isinstance(chunk, str):
                full_response += chunk
        
        # Add the response to chat history
        add_message_to_current_session("bot", full_response)
        
        # Clear streaming flags
        st.session_state.ai_responding = False
        st.session_state.last_user_message = None
        st.session_state.last_chat_messages = None
        
        # Save to history
        try:
            save_chat_history(st.session_state.user_id, get_current_session_messages())
        except:
            pass
        
        # Rerun to update the UI
        st.rerun()
        
    except Exception as e:
        # Clear streaming flags on error
        st.session_state.ai_responding = False
        st.session_state.last_user_message = None
        st.session_state.last_chat_messages = None
        
        # Add error message
        add_message_to_current_session("bot", f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")
        st.rerun()

# ---------- TAB 2: NOTES & QUIZ ----------
with tab2:
    # Update active tab
    st.session_state.active_tab = "Notes & Quiz"
    
    # FIXED ALIGNMENT: Clear separation of states using if-elif-else
    if st.session_state.completed and st.session_state.quiz:
        # STATE 3: Quiz Completed - Show Results
        st.markdown("---")
        # FIXED: Removed duplicate "Quiz Results" header
        st.subheader("🎉 Quiz Results")
        
        try:
            # Convert answers for compatibility
            answers_for_evaluation = {str(k): v for k, v in st.session_state.answers.items()}
            
            result = evaluate_quiz_attempt(
                st.session_state.quiz,
                answers_for_evaluation,
                st.session_state.user_id
            )
            
            # Score metrics with beautiful design
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(79, 222, 216, 0.1), rgba(0, 217, 255, 0.05)); border-radius: 12px;">
                    <h3 style="color: #4fded8; margin: 0;">Score</h3>
                    <h1 style="margin: 10px 0; font-size: 48px; background: linear-gradient(90deg, #4fded8, #8a7bff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {result['score']}/{result['total']}
                    </h1>
                    <p style="color: #9aa3bf; margin: 0;">Raw Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(138, 123, 255, 0.1), rgba(179, 102, 255, 0.05)); border-radius: 12px;">
                    <h3 style="color: #8a7bff; margin: 0;">Accuracy</h3>
                    <h1 style="margin: 10px 0; font-size: 48px; background: linear-gradient(90deg, #8a7bff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {result['accuracy']}%
                    </h1>
                    <p style="color: #9aa3bf; margin: 0;">Performance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                difficulty = result.get('difficulty', 'Medium').title()
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(255, 142, 107, 0.05)); border-radius: 12px;">
                    <h3 style="color: #ff6b9d; margin: 0;">Difficulty</h3>
                    <h1 style="margin: 10px 0; font-size: 32px; color: {'#4fded8' if difficulty == 'Easy' else '#8a7bff' if difficulty == 'Medium' else '#ff6b9d' if difficulty == 'Hard' else '#ff416c'}">
                        {difficulty}
                    </h1>
                    <p style="color: #9aa3bf; margin: 0;">Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            # FIXED: Question Type Analysis - Using separate HTML calls
            if st.session_state.difficulty == "Difficult":
                st.markdown("### 🔍 Question Type Analysis")
                
                # Calculate question type statistics
                single_count = 0
                multi_count = 0
                single_correct = 0
                multi_correct = 0
                
                for feedback in result.get("feedback", []):
                    if feedback.get("answer_type") == "single":
                        single_count += 1
                        if feedback.get("is_correct"):
                            single_correct += 1
                    elif feedback.get("answer_type") == "multiple":
                        multi_count += 1
                        if feedback.get("is_correct"):
                            multi_correct += 1
                
                col1, col2 = st.columns(2)
                with col1:
                    if single_count > 0:
                        single_acc = (single_correct / single_count * 100)
                        st.markdown(f"""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(79, 222, 216, 0.1), rgba(0, 217, 255, 0.05)); border-radius: 12px;">
                            <h4 style="color: #4fded8;">📝 Single-Answer Questions</h4>
                            <p style="font-size: 24px; font-weight: bold; color: #4fded8;">{single_correct}/{single_count}</p>
                            <p style="color: #9aa3bf;">Correct ({single_acc:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Determine scenario for single answer progress
                        scenario = "excellent" if single_acc >= 80 else "good" if single_acc >= 60 else "fair" if single_acc >= 40 else "needs_improvement"
                        st.markdown(create_gradient_progress_bar(single_acc, st.session_state.difficulty, scenario), unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(79, 222, 216, 0.1), rgba(0, 217, 255, 0.05)); border-radius: 12px;">
                            <h4 style="color: #4fded8;">📝 Single-Answer Questions</h4>
                            <p style="color: #9aa3bf;">No single-answer questions in this quiz</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if multi_count > 0:
                        multi_acc = (multi_correct / multi_count * 100)
                        st.markdown(f"""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(138, 123, 255, 0.1), rgba(179, 102, 255, 0.05)); border-radius: 12px;">
                            <h4 style="color: #8a7bff;">🔢 Multiple-Answer Questions</h4>
                            <p style="font-size: 24px; font-weight: bold; color: #8a7bff;">{multi_correct}/{multi_count}</p>
                            <p style="color: #9aa3bf;">Correct ({multi_acc:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Determine scenario for multiple answer progress
                        scenario = "excellent" if multi_acc >= 80 else "good" if multi_acc >= 60 else "fair" if multi_acc >= 40 else "needs_improvement"
                        st.markdown(create_gradient_progress_bar(multi_acc, st.session_state.difficulty, scenario), unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(138, 123, 255, 0.1), rgba(179, 102, 255, 0.05)); border-radius: 12px;">
                            <h4 style="color: #8a7bff;">🔢 Multiple-Answer Questions</h4>
                            <p style="color: #9aa3bf;">No multiple-answer questions in this quiz</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Partial credit info for multiple answer questions
            if result.get('partial_credit', 0) > 0:
                st.info(f"✨ You earned **{result['partial_credit']:.1f} points** in partial credit for partially correct multiple-answer questions!")
            
            # Detailed feedback
            with st.expander("📝 Review All Questions", expanded=False):
                for i, feedback in enumerate(result.get("feedback", [])):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {feedback.get('question', '')}")
                        
                        # Show answer type
                        answer_type = feedback.get('answer_type', 'single')
                        type_label = "SINGLE ANSWER" if answer_type == 'single' else "MULTIPLE ANSWERS"
                        type_color = "#4fded8" if answer_type == 'single' else "#8a7bff"
                        st.markdown(f'<span style="display: inline-block; padding: 4px 8px; background: {type_color}15; border: 1px solid {type_color}30; border-radius: 6px; font-size: 11px; font-weight: bold; color: {type_color}; margin-left: 10px;">{type_label}</span>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            status = feedback.get('status', '')
                            if "✅" in status:
                                st.success("✅ Correct")
                            elif "⚠️" in status or "Partially" in status:
                                st.warning("⚠️ Partially Correct")
                            else:
                                st.error("❌ Incorrect")
                        
                        with col2:
                            user_answers = feedback.get('user_answers', [])
                            correct_answers = feedback.get('correct_answers', [])
                            
                            if answer_type == 'multiple':
                                st.write(f"**Your Answers:** {', '.join(user_answers) if user_answers else 'None'}")
                                st.write(f"**Correct Answers:** {', '.join(correct_answers)}")
                            else:
                                st.write(f"**Your Answer:** {feedback.get('user_answers', [''])[0] if feedback.get('user_answers') else 'N/A'}")
                                st.write(f"**Correct Answer:** {feedback.get('correct_answers', [''])[0] if feedback.get('correct_answers') else 'N/A'}")
                            
                            st.write(f"**Explanation:** {feedback.get('explanation', '')}")
                        
                        st.markdown("---")
            
            # Action buttons
            st.markdown("### What would you like to do next?")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Take Another Quiz", use_container_width=True):
                    # Reset quiz state
                    st.session_state.quiz = None
                    st.session_state.completed = False
                    st.session_state.current_q = 0
                    st.session_state.answers = {}
                    st.session_state.selected_options = {}
                    st.session_state.question_answered = {}
                    st.session_state.show_feedback = {}
                    st.session_state.answer_submitted = {}
                    st.session_state.question_status = {}
                    st.session_state.last_clicked_option = None
                    st.session_state.attempt += 1
                    
                    st.session_state.need_rerun = True
            
            with col2:
                if st.button("📚 Study More", use_container_width=True):
                    # Switch to chat tab to ask questions
                    topic = st.session_state.quiz.get("topic", "this topic")
                    st.session_state.chat_input = f"Help me understand {topic} better. I scored {result['accuracy']}% on the quiz."
                    st.session_state.current_session_id = None
                    # Set tab state for navigation
                    st.session_state.active_tab = "Chat"
                    st.rerun()
            
            with col3:
                if st.button("📊 View Progress", use_container_width=True):
                    # Switch to progress tab
                    st.session_state.active_tab = "Progress"
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error evaluating quiz: {str(e)}")
    
    elif st.session_state.quiz and not st.session_state.completed:
        # STATE 2: Quiz Active - Show Only Quiz
        # Show quiz navigation header with back button
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("📝 Take the Quiz")
        with col_header2:
            if st.button("← Back to Setup", type="secondary", use_container_width=True):
                # Reset quiz state
                st.session_state.quiz = None
                st.session_state.completed = False
                st.session_state.current_q = 0
                st.session_state.answers = {}
                st.session_state.selected_options = {}
                st.session_state.question_answered = {}
                st.session_state.show_feedback = {}
                st.session_state.answer_submitted = {}
                st.session_state.question_status = {}
                st.session_state.last_clicked_option = None
                st.rerun()
        
        st.markdown("---")
        
        # Display the quiz
        quiz = st.session_state.quiz.get("quiz", [])
        total = len(quiz)
        idx = st.session_state.current_q
        
        if idx < total:
            q = quiz[idx]
            answer_type = q.get("answer_type", "single")
            options = q.get("options", {})
            correct_answer = q.get("answer", "").strip().upper()
            
            # Create a two-column layout for progress bar and circle progress
            col_progress, col_circle = st.columns([2, 1])
            
            with col_progress:
                # FIXED: Display progress indicator in separate parts
                progress_data = create_quiz_progress_indicator(idx, total, st.session_state.difficulty)
                
                # Display the container
                st.markdown('<div class="quiz-progress-container">', unsafe_allow_html=True)
                
                # Display header with dots legend
                st.markdown(progress_data["header"], unsafe_allow_html=True)
                
                # Display multi-segment progress bar
                st.markdown(progress_data["segment_bar"], unsafe_allow_html=True)
                
                # Display dots separately
                st.markdown(progress_data["dots"], unsafe_allow_html=True)
                
                # Close container
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_circle:
                # Calculate completed questions count
                completed_count = len([i for i in range(total) if i in st.session_state.question_answered])
                
                # Create circle progress bar with consistent gradient (FIXED)
                circle_html = create_circle_progress_bar(completed_count, total, idx)
                st.markdown(circle_html, unsafe_allow_html=True)
                
                # Status indicator below circle - shows progress only (FIXED)
                if completed_count > 0:
                    # Always use purple for consistency
                    status_color = "#8a7bff"
                    status_icon = "📊"
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px; font-size: 12px; color: {status_color};">
                        {status_icon} Progress: {completed_count}/{total} questions
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px; font-size: 12px; color: #9aa3bf;">
                        🚀 Start your quiz!
                    </div>
                    """, unsafe_allow_html=True)
            
            # Question box
            st.markdown("<div class='quiz-box'>", unsafe_allow_html=True)
            
            # Display the question with mode indicator
            mode_indicator = ""
            if st.session_state.difficulty != "Difficult":
                if answer_type == "multiple":
                    mode_indicator = '<span class="mode-indicator multi-mode">MULTIPLE ANSWERS</span>'
                else:
                    mode_indicator = '<span class="mode-indicator single-mode">SINGLE ANSWER</span>'
            else:
                mode_indicator = '<span class="mode-indicator" style="background:rgba(255,107,157,0.2);color:#ff6b9d;">TYPE: ???</span>'
            
            st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <span class='difficulty-badge difficulty-{st.session_state.difficulty.lower()}'>{st.session_state.difficulty.upper()}</span>
                {mode_indicator}
            </div>
            <div style='font-size: 22px; font-weight: 600; margin-bottom: 25px; color: #e6eef9; line-height: 1.4;'>{q['question']}</div>
            """, unsafe_allow_html=True)
            
            # Show instructions based on difficulty
            if st.session_state.difficulty != "Difficult":
                st.markdown(get_difficulty_instructions(st.session_state.difficulty, answer_type), unsafe_allow_html=True)
            else:
                st.markdown(get_difficulty_instructions(st.session_state.difficulty), unsafe_allow_html=True)
            
            # Check if question has been answered
            answered = idx in st.session_state.question_answered
            show_feedback = st.session_state.show_feedback.get(idx, False)
            
            if not answered or not show_feedback:
                # Initialize selected options for this question if not exists
                if idx not in st.session_state.selected_options:
                    st.session_state.selected_options[idx] = set()
                
                # Get current selection
                current_selection = st.session_state.selected_options[idx]
                
                # Show selection counter for multiple answers
                if len(current_selection) > 0:
                    if st.session_state.difficulty == "Difficult" or answer_type == "multiple":
                        st.markdown(f"<div style='margin-bottom: 15px;'><strong>Selected:</strong> <span class='selection-counter'>{len(current_selection)} option(s)</span></div>", unsafe_allow_html=True)
                
                # Display directly clickable option cards using the UPDATED function
                display_clickable_options(idx, options, current_selection, answer_type, st.session_state.difficulty)
                
                # Submit button
                current_selection = st.session_state.selected_options.get(idx, set())
                submit_disabled = len(current_selection) == 0
                
                # Modern submit button
                submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
                with submit_col2:
                    submit_label = "✅ Submit Answer & See Result"
                    submit_key = f"submit_{idx}"
                    
                    submit_btn = st.button(
                        submit_label,
                        key=submit_key,
                        use_container_width=True,
                        type="primary",
                        disabled=submit_disabled
                    )
                    
                    if submit_btn:
                        # Store the answer
                        if current_selection:
                            if answer_type == "single":
                                st.session_state.answers[idx] = list(current_selection)[0].upper()
                            else:
                                st.session_state.answers[idx] = ",".join(sorted([opt.upper() for opt in list(current_selection)]))
                        
                        # Show feedback for ALL modes
                        submit_answer(idx, answer_type, show_feedback=True)
                        
                        # Set rerun flag
                        st.session_state.need_rerun = True
            
            # Show feedback for ALL answered questions (ALL modes including Difficult)
            elif answered and show_feedback:
                st.markdown("---")
                user_answer = st.session_state.answers.get(idx, "")
                
                # Different header based on difficulty
                if st.session_state.difficulty == "Difficult":
                    st.markdown("### 🎯 Difficult Mode Result:")
                    st.markdown(f"""
                    <div style="padding: 10px; background: linear-gradient(135deg, rgba(255,107,157,0.1), rgba(255,65,108,0.05)); 
                                border-radius: 8px; margin: 10px 0; text-align: center;">
                        <span style="color: #ff6b9d; font-weight: bold;">
                        🔍 Question Type Revealed: {answer_type.upper().replace('_', ' ')}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("### 📊 Your Result:")
                
                # Display options with card-based feedback - One option per line
                st.markdown('<div class="option-grid">', unsafe_allow_html=True)
                
                for option_key, option_text in options.items():
                    option_upper = option_key.upper()
                    css_class = "option-card-interactive"
                    icon = option_key  # Default to option letter
                    
                    if answer_type == "single":
                        if option_upper == correct_answer:
                            css_class += " correct"  # Correct answer (always show in green)
                            icon = "✅"
                            hint = "Correct Answer"
                        elif option_upper == user_answer.upper():
                            css_class += " wrong"    # User selected wrong answer
                            icon = "❌"
                            hint = "Your Selection (Incorrect)"
                        else:
                            hint = "Not Selected"
                    else:
                        # Multiple answer question
                        user_answers_set = set([ans.strip().upper() for ans in user_answer.split(",")]) if user_answer else set()
                        correct_answers_set = set([ans.strip().upper() for ans in correct_answer.split(",")])
                        
                        if option_upper in correct_answers_set and option_upper in user_answers_set:
                            css_class += " correct"  # Correctly selected
                            icon = "✅"
                            hint = "Correct (You selected this)"
                        elif option_upper in correct_answers_set and option_upper not in user_answers_set:
                            css_class += " wrong"    # Missed correct answer
                            icon = "⚠️"
                            hint = "Correct Answer (You missed this)"
                        elif option_upper not in correct_answers_set and option_upper in user_answers_set:
                            css_class += " wrong"    # Incorrectly selected
                            icon = "❌"
                            hint = "Incorrect (You selected this)"
                        else:
                            hint = "Not Selected"
                    
                    # Create visual feedback card
                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="display: flex; align-items: center;">
                            <div class="option-label-circle">{icon}</div>
                            <div class="option-content">
                                <div class="option-text">{option_text}</div>
                                <div class="option-hint">{hint}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show summary message
                if answer_type == "single":
                    if user_answer.upper() == correct_answer:
                        st.success("### ✅ **Correct!** Well done!")
                        st.session_state.question_status[idx] = "correct"
                    else:
                        st.error(f"### ❌ **Incorrect.** The correct answer is **{correct_answer}**")
                        st.session_state.question_status[idx] = "incorrect"
                else:
                    user_answers_set = set([ans.strip().upper() for ans in user_answer.split(",")]) if user_answer else set()
                    correct_answers_set = set([ans.strip().upper() for ans in correct_answer.split(",")])
                    
                    if user_answers_set:
                        st.write(f"**Your selected answers:** {', '.join(sorted(user_answers_set))}")
                    
                    if user_answers_set == correct_answers_set:
                        st.success(f"### ✅ **Perfect!** All {len(correct_answers_set)} answers correct!")
                        st.session_state.question_status[idx] = "correct"
                    else:
                        correct_selected = len(user_answers_set.intersection(correct_answers_set))
                        total_correct = len(correct_answers_set)
                        
                        if correct_selected > 0:
                            st.warning(f"### ⚠️ **Partially correct:** You got {correct_selected}/{total_correct} correct")
                            st.info(f"**All correct answers:** {', '.join(sorted(correct_answers_set))}")
                            st.session_state.question_status[idx] = "partial"
                        else:
                            st.error(f"### ❌ **No correct answers selected.**")
                            st.info(f"**Correct answers:** {', '.join(sorted(correct_answers_set))}")
                            st.session_state.question_status[idx] = "incorrect"
                
                # Show explanation if available
                if q.get('explanation'):
                    with st.expander("📖 Explanation", expanded=True):
                        st.info(q['explanation'])
                
                # Navigation buttons
                st.markdown("<div class='nav-btn-container'>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if idx > 0:
                        if st.button("← Previous Question", key=f"prev_feedback_{idx}", use_container_width=True):
                            st.session_state.current_q -= 1
                            st.session_state.need_rerun = True
                
                with col2:
                    if idx + 1 < total:
                        if st.button("Next Question →", key=f"next_feedback_{idx}", use_container_width=True, type="primary"):
                            st.session_state.current_q += 1
                            st.session_state.need_rerun = True
                    else:
                        if st.button("Finish Quiz", key="finish_quiz_feedback", type="primary", use_container_width=True):
                            st.session_state.completed = True
                            st.session_state.need_rerun = True
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # STATE 1: No Quiz - Show Setup Interface
        st.subheader("📘 Upload Notes or Enter Topic for Quiz")
        
        # Quiz configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.quiz_source = st.radio(
                "Generate quiz from:", ["Notes", "Topic"], horizontal=True
            )
        with col2:
            st.session_state.difficulty = st.selectbox(
                "Difficulty:", 
                ["Easy", "Medium", "Hard", "Difficult"],
                help="Difficult: Multiple correct answers, tricky questions"
            )
        with col3:
            st.session_state.num_questions = st.selectbox("Questions:", [5, 10, 15])
        
        # Show warning for Difficult level
        if st.session_state.difficulty == "Difficult":
            st.markdown(
                '<div class="difficult-warning-box">'
                '⚠️ <strong>Difficult Mode:</strong> Questions may have single OR multiple correct answers! '
                'You won\'t know which type until submission. Full feedback shown after submission.'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Notes upload or topic input
        if st.session_state.quiz_source == "Notes":
            st.info("📤 Upload your study notes (PDF, DOCX, or TXT)")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "txt"],
                key="file_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Clear previous notes if new file is uploaded
                if uploaded_file != st.session_state.get("uploaded_file"):
                    st.session_state.notes_text = ""
                    st.session_state.uploaded_file = uploaded_file
                
                # Extract text from file
                if not st.session_state.notes_text:
                    # Show custom loader for text extraction with reduced gap
                    loader_placeholder = st.empty()
                    with loader_placeholder.container():
                        st.markdown(show_custom_loader(f"Extracting text from {uploaded_file.name}..."), unsafe_allow_html=True)
                    
                    suffix = os.path.splitext(uploaded_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    try:
                        st.session_state.notes_text = extract_text(temp_path)
                        loader_placeholder.empty()  # Clear loader
                        
                        if st.session_state.notes_text:
                            st.success(f"✅ Successfully extracted {len(st.session_state.notes_text)} characters from {uploaded_file.name}")
                            
                            # Show preview
                            with st.expander("📄 Preview extracted notes", expanded=True):
                                preview_text = st.session_state.notes_text[:1500]
                                if len(st.session_state.notes_text) > 1500:
                                    preview_text += "..."
                                st.text_area(
                                    "Notes Content",
                                    preview_text,
                                    height=200,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                            
                            # Store notes in vector database
                            try:
                                store_notes_and_progress(
                                    user_id=st.session_state.user_id,
                                    notes_text=st.session_state.notes_text
                                )
                                st.info("📚 Notes saved to your knowledge base!")
                            except Exception as e:
                                st.warning(f"Note: Could not save to database: {str(e)}")
                                
                    except Exception as e:
                        loader_placeholder.empty()  # Clear loader on error
                        st.error(f"❌ Error extracting text: {str(e)}")
                    finally:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                else:
                    # Show already extracted notes
                    st.success(f"✅ Notes already extracted from {uploaded_file.name}")
                    
                    with st.expander("📄 View extracted notes", expanded=False):
                        preview_text = st.session_state.notes_text[:1500]
                        if len(st.session_state.notes_text) > 1500:
                            preview_text += "..."
                        st.text_area(
                            "Notes Content",
                            preview_text,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
            
            else:
                # No file uploaded yet
                st.markdown("""
                <div style="text-align: center; padding: 40px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px; margin: 20px 0;">
                    <h3 style="color: #8a7bff;">📤 Upload Your Notes</h3>
                    <p>Drag and drop or click to upload PDF, DOCX, or TXT files</p>
                    <p style="font-size: 12px; color: #9aa3bf;">Supported formats: .pdf, .docx, .txt</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Topic mode
            st.info("🎯 Enter a topic to generate a quiz")
            
            st.session_state.custom_topic = st.text_input(
                "Enter your topic:",
                placeholder="e.g., Machine Learning, World History, Biology, Python Programming, etc.",
                value=st.session_state.custom_topic,
                key="topic_input"
            )
            
            if st.session_state.custom_topic:
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(138, 123, 255, 0.1); border-radius: 8px; margin: 10px 0;">
                    <strong>Topic:</strong> {st.session_state.custom_topic}<br>
                    <small style="color: #9aa3bf;">Will generate {st.session_state.num_questions} {st.session_state.difficulty.lower()} questions</small>
                    {"<br><small style='color: #ff6b6b;'>⚠️ Difficult mode: Question types hidden until submission</small>" if st.session_state.difficulty == "Difficult" else ""}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generate quiz button with loader placeholder above it
        generate_disabled = False
        generate_tooltip = ""
        
        if st.session_state.quiz_source == "Notes":
            if not st.session_state.notes_text:
                generate_disabled = True
                generate_tooltip = "Please upload and extract notes first"
        else:  # Topic mode
            if not st.session_state.custom_topic or not st.session_state.custom_topic.strip():
                generate_disabled = True
                generate_tooltip = "Please enter a topic first"
        
        # Loader placeholder for quiz generation
        quiz_loader_placeholder = st.empty()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_clicked = st.button(
                "🚀 Generate Quiz",
                disabled=generate_disabled,
                use_container_width=True,
                type="primary",
                help=generate_tooltip if generate_tooltip else None
            )
        
        if generate_clicked:
            # Show custom loader for quiz generation with reduced gap
            with quiz_loader_placeholder.container():
                loader_text = f"Generating {st.session_state.num_questions} {st.session_state.difficulty} questions..."
                st.markdown(show_custom_loader(loader_text), unsafe_allow_html=True)
            
            try:
                # Prepare input text
                if st.session_state.quiz_source == "Notes":
                    input_text = st.session_state.notes_text
                    source_desc = "from your notes"
                else:
                    input_text = st.session_state.custom_topic
                    source_desc = f"about {st.session_state.custom_topic}"
                
                # Generate quiz
                quiz_data = generate_quiz_from_notes(
                    notes_text=input_text,
                    user_id=st.session_state.user_id,
                    num_questions=st.session_state.num_questions,
                    difficulty=st.session_state.difficulty.lower(),
                )
                
                quiz_loader_placeholder.empty()  # Clear loader
                
                if quiz_data and quiz_data.get("quiz"):
                    st.session_state.quiz = quiz_data
                    st.session_state.current_q = 0
                    st.session_state.completed = False
                    st.session_state.answers = {}
                    st.session_state.selected_options = {}
                    st.session_state.question_answered = {}
                    st.session_state.show_feedback = {}
                    st.session_state.answer_submitted = {}
                    st.session_state.question_status = {}
                    st.session_state.last_clicked_option = None
                    
                    st.success(f"✅ Successfully generated {len(quiz_data['quiz'])} questions {source_desc}!")
                    
                    # Show quiz statistics
                    single_count = sum(1 for q in quiz_data['quiz'] if q.get('answer_type') == 'single')
                    multi_count = sum(1 for q in quiz_data['quiz'] if q.get('answer_type') == 'multiple')
                    
                    if multi_count > 0 and st.session_state.difficulty != "Difficult":
                        st.info(f"📊 Quiz includes: {single_count} single-answer and {multi_count} multiple-answer questions")
                    elif st.session_state.difficulty == "Difficult":
                        st.info("🎯 Difficult Mode: Question types are hidden until submission!")
                    
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to generate quiz. Please try again with different content.")
                    
            except Exception as e:
                quiz_loader_placeholder.empty()  # Clear loader on error
                st.error(f"❌ Error generating quiz: {str(e)}")

# ---------- TAB 3: PROGRESS ----------
with tab3:
    # Update active tab
    st.session_state.active_tab = "Progress"
    
    st.subheader("📈 Your Learning Progress")
    
    if st.button("🔄 Refresh Progress", use_container_width=True, type="primary"):
        # Show custom loader for progress refresh with reduced gap
        loader_placeholder = st.empty()
        with loader_placeholder.container():
            st.markdown(show_custom_loader("Refreshing progress..."), unsafe_allow_html=True)
        
        try:
            st.rerun()
        finally:
            loader_placeholder.empty()  # Clear loader
    
    try:
        data = fetch_progress_from_pinecone(st.session_state.user_id)
        
        if data and data.get("progress"):
            summary = data.get("summary", {})
            progress_list = data.get("progress", [])
            
            if report_icon:
                report_icon = f'<img src="data:image/png;base64,{report_icon}" class="report-icon"> '
                unsafe_allow_html=True
            else:
                report_icon= "📊 "
            # Summary metrics with beautiful design - ALL EQUALLY SIZED
            st.markdown(f"### {report_icon} Overall Statistics", unsafe_allow_html=True)
            
            # Use equal columns for all metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Total Attempts
            with col1:
                total_attempts = summary.get("total_attempts", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📚</div>
                    <div class="metric-label">Total Attempts</div>
                    <div class="metric-value">{total_attempts}</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Quizzes Taken</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Average Score
            with col2:
                avg_score = summary.get('average_score', 0)
                scenario = "excellent" if avg_score >= 80 else "good" if avg_score >= 60 else "fair" if avg_score >= 40 else "needs_improvement"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-label">Average Score</div>
                    <div class="metric-value">{avg_score:.1f}</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Per Quiz</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(create_gradient_progress_bar(avg_score, "medium", scenario), unsafe_allow_html=True)
            
            # Average Accuracy
            with col3:
                avg_acc = summary.get('average_accuracy', 0)
                scenario = "excellent" if avg_acc >= 80 else "good" if avg_acc >= 60 else "fair" if avg_acc >= 40 else "needs_improvement"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-label">Average Accuracy</div>
                    <div class="metric-value">{avg_acc:.1f}%</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Performance</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(create_gradient_progress_bar(avg_acc, "hard", scenario), unsafe_allow_html=True)
            
            # Topics Covered
            with col4:
                topics = summary.get("topics_covered", [])
                topic_count = len(topics)
                topic_percentage = min(100, (topic_count / 10) * 100) if topics else 0
                scenario = "excellent" if topic_count >= 8 else "good" if topic_count >= 5 else "fair" if topic_count >= 3 else "needs_improvement"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📖</div>
                    <div class="metric-label">Topics Covered</div>
                    <div class="metric-value">{topic_count}</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Areas Studied</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(create_gradient_progress_bar(topic_percentage, "difficult", scenario), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Topics covered in a grid
            if topics:
                st.subheader("📚 Topics You've Studied")
                cols = st.columns(3)
                for i, topic in enumerate(topics[:9]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(138, 123, 255, 0.1), rgba(179, 102, 255, 0.05)); 
                                    border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #8a7bff;">
                            <p style="margin: 0; font-weight: 500; color: #e6eef9;">{topic}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Progress chart
            st.subheader("📋 Attempt History")
            
            for i, attempt in enumerate(progress_list[:10]):
                accuracy = attempt.get('accuracy', 0)
                score = attempt.get('score', 0)
                total = attempt.get('total', 1)
                calendar_icon_html = "",
                if Calendar_Icon:
                    calendar_icon_html = f'![calendar-icon-class](data:image/png;base64,{Calendar_Icon}) ' 
                    
                    
                else:
                    calendar_icon_html = "📅 "
                
                with st.expander(f"{calendar_icon_html} {time.strftime('%Y-%m-%d %H:%M', time.localtime(attempt.get('timestamp', time.time())))}, Score: {score}/{total} , Accuracy: {accuracy}%", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Topic:** {attempt.get('topic', 'Unknown Topic')}")
                        st.write(f"**Difficulty:** {attempt.get('difficulty', 'Medium').title()}")
                        st.write(f"**Source:** {attempt.get('source', 'Unknown')}")
                        
                    with col2:
                        # Progress bar for accuracy with scenario-based gradient
                        st.markdown("**Accuracy:**")
                        scenario = "excellent" if accuracy >= 80 else "good" if accuracy >= 60 else "fair" if accuracy >= 40 else "needs_improvement"
                        difficulty = attempt.get('difficulty', 'medium').lower()
                        st.markdown(create_gradient_progress_bar(accuracy, difficulty, scenario), unsafe_allow_html=True)
                        
                        # Show question type breakdown if available
                        single_correct = 0
                        single_total = 0
                        multi_correct = 0
                        multi_total = 0
                        
                        for fb in attempt.get("feedback", []):
                            if fb.get("answer_type") == "single":
                                single_total += 1
                                if fb.get("is_correct"):
                                    single_correct += 1
                            elif fb.get("answer_type") == "multiple":
                                multi_total += 1
                                if fb.get("is_correct"):
                                    multi_correct += 1
                        
                        if single_total > 0:
                            single_acc = (single_correct / single_total) * 100
                            scenario = "excellent" if single_acc >= 80 else "good" if single_acc >= 60 else "fair" if single_acc >= 40 else "needs_improvement"
                            st.write(f"📝 **Single:** {single_correct}/{single_total} ({single_acc:.1f}%)")
                            st.markdown(create_gradient_progress_bar(single_acc, "easy", scenario), unsafe_allow_html=True)
                        
                        if multi_total > 0:
                            multi_acc = (multi_correct / multi_total) * 100
                            scenario = "excellent" if multi_acc >= 80 else "good" if multi_acc >= 60 else "fair" if multi_acc >= 40 else "needs_improvement"
                            st.write(f"🔢 **Multiple:** {multi_correct}/{multi_total} ({multi_acc:.1f}%)")
                            st.markdown(create_gradient_progress_bar(multi_acc, "medium", scenario), unsafe_allow_html=True)
        
        else:
            st.info("📊 No progress data found yet!")
            st.markdown("""
            <div style="padding: 30px; background: linear-gradient(135deg, rgba(79, 222, 216, 0.05), rgba(138, 123, 255, 0.05)); border-radius: 12px; text-align: center;">
                <h3 style="color: #8a7bff;">Start Your Learning Journey! 🚀</h3>
                <p>Complete quizzes to track your progress here.</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error loading progress: {str(e)}")

# -----------------------------
# Rerun logic at the end
# -----------------------------
if st.session_state.get("need_rerun", False):
    st.session_state.need_rerun = False
    st.rerun()