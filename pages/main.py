# pages/app.py - Fixed duplicate saving
import streamlit as st
import os
import tempfile
import base64
import time
import streamlit.components.v1 as components
from Notes_Quiz_Section import (
    generate_quiz_from_notes,
    generate_quiz_from_topic,
    evaluate_quiz_attempt,
    format_quiz_for_display,
    extract_text,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    store_notes_and_quizzes
)
from Chatbot import (
    retrieve_context,
    store_conversation,
    get_user_history,
    get_gemini_response,
    get_conversation_context,
)
from Progress import (
    store_progress,
    fetch_progress_from_pinecone,
    cleanup_duplicate_progress,
    format_progress_for_display
)

# -----------------------------
# CHECK LOGIN STATUS
# -----------------------------
if not st.session_state.get('logged_in', False) or not st.session_state.get('user_id'):
    st.switch_page("Login_Signup.py")
    st.stop()

# -----------------------------
# LOAD CSS FROM EXTERNAL FILE
# -----------------------------
def load_css():
    """Load CSS from external file"""
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback to minimal CSS if file not found
        st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background: linear-gradient(180deg, #061022, #0b1523) !important;
                color: #e6eef9 !important;
            }
        </style>
        """, unsafe_allow_html=True)

# Call the CSS loader
load_css()

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
Avatar_Icon = get_base64_file("Assets/Bot_Avatar.png") if os.path.exists("Assets/Bot_Avatar.png") else ""
tab_chat_icon = get_base64_file("Assets/tab_chat_icon.png") if os.path.exists("Assets/tab_chat_icon.png") else ""
Spinner_Loader = get_base64_file("Assets/Spinner_Loader.gif") if os.path.exists("Assets/Spinner_Loader.gif") else ""
notes_icon = get_base64_file("Assets/notes.png") if os.path.exists("Assets/notes.png") else ""
progress_icon = get_base64_file("Assets/progress.png") if os.path.exists("Assets/progress.png") else ""
report_icon = get_base64_file("Assets/3d-report.png") if os.path.exists("Assets/3d-report.png") else ""
Calendar_Icon = get_base64_file("Assets/Calendar.png") if os.path.exists("Assets/Calendar.png") else ""
target_icon = get_base64_file("Assets/target.png") if os.path.exists("Assets/target.png") else ""
book_icon = get_base64_file("Assets/book.jpeg") if os.path.exists("Assets/book.jpeg") else ""
quiz_icon = get_base64_file("Assets/quiz.jpeg") if os.path.exists("Assets/quiz.jpeg") else ""
bin_icon = get_base64_file("Assets/bin.png") if os.path.exists("Assets/bin.png") else ""
search_icon= get_base64_file("Assets/search_icon.png") if os.path.exists("Assets/search_icon.png") else ""

def show_custom_loader(text="Processing..."):
    """Display custom GIF loader using base64 with minimal gap"""
    if Spinner_Loader:
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
    "selected_options": {},
    "question_answered": {},
    "show_feedback": {},
    "answer_submitted": {},
    "question_status": {},
    "last_clicked_option": None,
    "need_rerun": False,
    "active_tab": "Notes & Quiz",
    "ai_responding": False,
    "streaming_message": "",
    "streaming_finished": False,
    "current_accuracy": 0,
    "chat_placeholder": None,
    "streaming_chunks": [],
    "last_user_message": None,
    "last_chat_messages": None,
    "_full_response": "",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# CONVERSATION SAVING WITH USER ID
# -----------------------------

def save_conversation_to_pinecone(user_id: str, question: str, answer: str, contexts: list = None):
    """Save conversation with user_id for retrieval"""
    try:
        store_conversation(
            user_id=user_id,
            question=question,
            answer=answer,
            contexts=contexts or []
        )
        print(f"✅ Conversation saved for user: {user_id}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save conversation: {e}")
        return False

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
                st.session_state.answers[question_idx] = list(selected)[0].upper()
            else:
                st.session_state.answers[question_idx] = ",".join(sorted([opt.upper() for opt in list(selected)]))
            
            st.session_state.question_answered[question_idx] = True
            st.session_state.answer_submitted[question_idx] = True
            st.session_state.show_feedback[question_idx] = True
            
            if question_idx in st.session_state.selected_options:
                del st.session_state.selected_options[question_idx]
            st.session_state.last_clicked_option = None
            st.session_state.need_rerun = True
        else:
            st.session_state.need_rerun = True
    else:
        st.session_state.need_rerun = True

# -----------------------------
# Quiz Progress Helpers
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
            if status == "correct":
                segment_class = "segment-correct"
            elif status == "partial":
                segment_class = "segment-partial"
            elif status == "incorrect":
                segment_class = "segment-incorrect"
            else:
                segment_class = "segment-pending"
        elif i == st.session_state.current_q:
            if status:
                if status == "correct":
                    segment_class = "segment-correct"
                elif status == "partial":
                    segment_class = "segment-partial"
                elif status == "incorrect":
                    segment_class = "segment-incorrect"
                else:
                    segment_class = "segment-pending"
            else:
                segment_class = "segment-active"
        else:
            segment_class = "segment-pending"
        
        segments_html += f'<div class="progress-segment {segment_class}" style="width: {segment_width}%;"></div>'
    
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

def get_current_accuracy():
    """Calculate current accuracy based on answered questions"""
    answered_questions = [i for i in st.session_state.question_answered.keys()]
    
    if not answered_questions:
        return 0
    
    correct_count = sum(1 for i in answered_questions if st.session_state.question_status.get(i) == "correct")
    partial_count = sum(1 for i in answered_questions if st.session_state.question_status.get(i) == "partial")
    
    weighted_correct = correct_count + (partial_count * 0.5)
    total_weighted = len(answered_questions)
    
    if total_weighted > 0:
        accuracy = (weighted_correct / total_weighted) * 100
    else:
        accuracy = 0
    
    return accuracy

def create_gradient_progress_bar(percentage, difficulty="medium", scenario="default"):
    """Create a beautiful gradient progress bar based on scenario"""
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
    accuracy = get_current_accuracy()
    
    dots_html = ""
    for i in range(total_questions):
        status_class = ""
        if i < current_idx:
            status = st.session_state.question_status.get(i, "")
            if status == "correct":
                status_class = "correct"
            elif status == "partial":
                status_class = "partial"
            elif status == "incorrect":
                status_class = "incorrect"
        elif i == current_idx:
            status_class = "active"
        
        dots_html += f'<div class="progress-dot {status_class}"></div>'
    
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
        "segment_bar": segment_bar,
        "dots": f'<div class="quiz-progress-dots">{dots_html}</div>',
        "accuracy": accuracy
    }

def create_circle_progress_bar(completed_count, total_questions, current_question_idx):
    """Create a circle progress bar"""
    progress_percentage = (completed_count / total_questions * 100) if total_questions > 0 else 0
    circle_gradient = "url(#circleGradient)"
    radius = 45
    circumference = 2 * 3.14159 * radius
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

def display_option_buttons(question_idx, options, current_selection, answer_type, difficulty):
    """Display option buttons"""
    if question_idx not in st.session_state.selected_options:
        st.session_state.selected_options[question_idx] = set(current_selection)
    
    container = st.container()
    
    with container:
        for option_key, option_text in options.items():
            is_selected = option_key in st.session_state.selected_options[question_idx]
            
            button_type = "primary" if is_selected else "secondary"
            button_label = f"✓ {option_key}. {option_text}" if is_selected else f"{option_key}. {option_text}"
            
            if st.button(button_label,
                        key=f"btn_{question_idx}_{option_key}_{is_selected}",
                        type=button_type,
                        use_container_width=True):
                current = st.session_state.selected_options[question_idx]
                
                if difficulty == "Difficult" or answer_type == "multiple":
                    if option_key in current:
                        current.remove(option_key)
                    else:
                        current.add(option_key)
                else:
                    st.session_state.selected_options[question_idx] = {option_key}
                
                st.session_state.last_clicked_option = (question_idx, option_key)
                st.session_state.need_rerun = True
                st.rerun()

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    f"""<div class="header-card"><div class="header-inner">
       <div class="header-avatar">
         <img src="data:image/png;base64,{Avatar_Icon if Avatar_Icon else 'https://cdn-icons-png.flaticon.com/512/4712/4712109.png'}">
       </div>
       <div><h1>Meet StudyBuddy</h1>
         <p>Your Personal AI Learning Assistant with Advanced Quiz Features</p>
         <p style="font-size: 12px; color: #8a7bff;">👤 Logged in as: {st.session_state.user_email}</p>
       </div>
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
    st.markdown(f"**User ID:** {st.session_state.user_id}")
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
        if bin_icon:
            bin_icon_html = f'![bin_icon-class](data:image/png;base64,{bin_icon})'
        else:
            bin_icon_html = "🗑️"
        if st.button(f"{bin_icon_html} Clear All Chats", use_container_width=True, type="secondary"):
            st.session_state.chat_sessions = []
            st.session_state.current_session_id = None
            st.rerun()
    else:
        st.info("No chat sessions yet!")

# -----------------------------
# MAIN TABS WITH ICONS
# -----------------------------
if tab_chat_icon:
    chat_icon = f"![Icon](data:image/png;base64,{tab_chat_icon}) "
else:
    chat_icon = "💬"
if notes_icon:
    notes_icon = f"![Icon](data:image/png;base64,{notes_icon}) "
else:
    notes_icon = "📝"
if progress_icon:
    progress_icon = f"![Icon](data:image/png;base64,{progress_icon}) "
else:
    progress_icon = "📊"

tab1, tab2, tab3 = st.tabs([
    f"{chat_icon} Chat",
    f"{notes_icon} Notes & Quiz", 
    f"{progress_icon} Progress"
])

# ---------- TAB 1: CHAT ----------
with tab1:
    st.session_state.active_tab = "Chat"
    
    def scroll_to_bottom():
        scroll_js = """
        <script>
        function scrollToBottom() {
            const container = document.querySelector('[data-testid="stVerticalBlock"]');
            if (container) {
                container.scrollTop = container.scrollHeight;
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
            }
            const appContainer = document.querySelector('[data-testid="stAppViewContainer"]');
            if (appContainer) {
                appContainer.scrollTop = appContainer.scrollHeight;
                appContainer.scrollTo({
                    top: appContainer.scrollHeight,
                    behavior: 'smooth'
                });
            }
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
        }
        scrollToBottom();
        setTimeout(scrollToBottom, 100);
        setTimeout(scrollToBottom, 300);
        setTimeout(scrollToBottom, 500);
        setTimeout(scrollToBottom, 1000);
        </script>
        """
        st.markdown(scroll_js, unsafe_allow_html=True)
    
    current_msgs = get_current_session_messages()
    
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
                st.markdown(
                    f'<div style="text-align:left; margin:10px 0;"><div class="msg-bubble msg-bot">'
                    f"🤖 {txt}<span class='msg-time'>{tstr}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        
        scroll_to_bottom()
        
        # ============================================
        # HANDLE STREAMING RESPONSE (FIXED - No Duplicate)
        # ============================================
        if st.session_state.get("ai_responding", False) and st.session_state.get("last_user_message"):
            try:
                response_stream = get_gemini_response(
                    st.session_state.last_user_message, 
                    st.session_state.last_chat_messages, 
                    st.session_state.user_id
                )
                
                # Store the question for saving
                user_question = st.session_state.last_user_message
                
                # Create generator that streams and saves ONCE
                def response_generator():
                    full_response = ""
                    for chunk in response_stream:
                        if chunk:
                            if hasattr(chunk, 'content'):
                                chunk_text = chunk.content
                            elif hasattr(chunk, 'text'):
                                chunk_text = chunk.text
                            elif isinstance(chunk, str):
                                chunk_text = chunk
                            else:
                                chunk_text = str(chunk)
                            
                            full_response += chunk_text
                            st.session_state.streaming_message = full_response
                            yield chunk_text
                            time.sleep(0.05)
                    
                    # SAVE ONCE HERE - after streaming completes
                    if full_response and user_question:
                        # Add to session messages
                        add_message_to_current_session("bot", full_response)
                        
                        # Save to Pinecone with user_id (ONCE)
                        save_conversation_to_pinecone(
                            user_id=st.session_state.user_id,
                            question=user_question,
                            answer=full_response,
                            contexts=[]
                        )
                        print(f"✅ Saved conversation for user: {st.session_state.user_id}")
                    
                    # Store full response for any cleanup
                    st.session_state._full_response = full_response
                
                # Display the stream
                st.write_stream(response_generator)
                scroll_to_bottom()
                
                # Clear streaming flags
                st.session_state.ai_responding = False
                st.session_state.streaming_message = ""
                st.session_state.last_user_message = None
                st.session_state.last_chat_messages = None
                st.session_state._full_response = ""
                
                st.rerun()
                
            except Exception as e:
                # Clear streaming flags on error
                st.session_state.ai_responding = False
                st.session_state.streaming_message = ""
                st.session_state.last_user_message = None
                st.session_state.last_chat_messages = None
                st.session_state._full_response = ""
                
                error_msg = f"Error: {str(e)}"
                add_message_to_current_session("bot", error_msg)
                st.error(f"❌ {error_msg}")
                st.rerun()
    
    user_msg = st.chat_input(
        "Ask me anything about your studies...",
        disabled=st.session_state.get("ai_responding", False),
        key="chat_input_native"
    )
    
    if user_msg:
        if not st.session_state.current_session_id:
            create_new_chat_session(user_msg)
        else:
            add_message_to_current_session("user", user_msg)
        
        st.session_state.ai_responding = True
        st.session_state.streaming_message = ""
        st.session_state.last_user_message = user_msg
        st.session_state.last_chat_messages = get_current_session_messages().copy()
        
        st.rerun()
    
    scroll_to_bottom()



# ---------- TAB 2: NOTES & QUIZ ----------
with tab2:
    st.session_state.active_tab = "Notes & Quiz"
    
    # STATE 3: Quiz Completed
    if st.session_state.completed and st.session_state.quiz:
        st.markdown("---")
        st.subheader("🎉 Quiz Results")
        
        try:
            answers_for_evaluation = {str(k): v for k, v in st.session_state.answers.items()}
            
            result = evaluate_quiz_attempt(
                st.session_state.quiz,
                answers_for_evaluation,
                st.session_state.user_id
            )
            
            # Score metrics
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
            
            # Question type analysis for Difficult mode
            if st.session_state.difficulty == "Difficult":
                st.markdown("### 🔍 Question Type Analysis")
                
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
            
            # Partial credit info
            if result.get('partial_credit', 0) > 0:
                st.info(f"✨ You earned **{result['partial_credit']:.1f} points** in partial credit for partially correct multiple-answer questions!")
            
            # Detailed feedback
            with st.expander("📝 Review All Questions", expanded=False):
                for i, feedback in enumerate(result.get("feedback", [])):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {feedback.get('question', '')}")
                        
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
                    topic = st.session_state.quiz.get("topic", "this topic")
                    st.session_state.chat_input = f"Help me understand {topic} better. I scored {result['accuracy']}% on the quiz."
                    st.session_state.current_session_id = None
                    st.session_state.active_tab = "Chat"
                    st.rerun()
            
            with col3:
                if st.button("📊 View Progress", use_container_width=True):
                    st.session_state.active_tab = "Progress"
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error evaluating quiz: {str(e)}")
    
    # STATE 2: Quiz Active
    elif st.session_state.quiz and not st.session_state.completed:
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            if quiz_icon:
                quiz_icon_html = f'![quiz-icon-class](data:image/png;base64,{quiz_icon}) '
            else:
                quiz_icon_html = "📝"
            st.subheader(quiz_icon_html + "Take the Quiz")
        with col_header2:
            if st.button("← Back to Setup", type="secondary", use_container_width=True):
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
        
        quiz = st.session_state.quiz.get("quiz", [])
        total = len(quiz)
        idx = st.session_state.current_q
        
        if idx < total:
            q = quiz[idx]
            answer_type = q.get("answer_type", "single")
            options = q.get("options", {})
            correct_answer = q.get("answer", "").strip().upper()
            
            col_progress, col_circle = st.columns([2, 1])
            
            with col_progress:
                progress_data = create_quiz_progress_indicator(idx, total, st.session_state.difficulty)
                
                st.markdown('<div class="quiz-progress-container">', unsafe_allow_html=True)
                st.markdown(progress_data["header"], unsafe_allow_html=True)
                st.markdown(progress_data["segment_bar"], unsafe_allow_html=True)
                st.markdown(progress_data["dots"], unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_circle:
                completed_count = len([i for i in range(total) if i in st.session_state.question_answered])
                circle_html = create_circle_progress_bar(completed_count, total, idx)
                st.markdown(circle_html, unsafe_allow_html=True)
                
                if completed_count > 0:
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px; font-size: 12px; color: #8a7bff;">
                        📊 Progress: {completed_count}/{total} questions
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px; font-size: 12px; color: #9aa3bf;">
                        🚀 Start your quiz!
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<div class='quiz-box'>", unsafe_allow_html=True)
            
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
            
            st.markdown(get_difficulty_instructions(st.session_state.difficulty, answer_type), unsafe_allow_html=True)
            
            answered = idx in st.session_state.question_answered
            show_feedback = st.session_state.show_feedback.get(idx, False)
            
            if not answered or not show_feedback:
                if idx not in st.session_state.selected_options:
                    st.session_state.selected_options[idx] = set()
                
                current_selection = st.session_state.selected_options[idx]
                
                if len(current_selection) > 0:
                    if st.session_state.difficulty == "Difficult" or answer_type == "multiple":
                        st.markdown(f"<div style='margin-bottom: 15px;'><strong>Selected:</strong> <span class='selection-counter'>{len(current_selection)} option(s)</span></div>", unsafe_allow_html=True)
                
                display_option_buttons(idx, options, current_selection, answer_type, st.session_state.difficulty)
                
                current_selection = st.session_state.selected_options.get(idx, set())
                submit_disabled = len(current_selection) == 0
                
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
                        if current_selection:
                            if answer_type == "single":
                                st.session_state.answers[idx] = list(current_selection)[0].upper()
                            else:
                                st.session_state.answers[idx] = ",".join(sorted([opt.upper() for opt in list(current_selection)]))
                        
                        submit_answer(idx, answer_type, show_feedback=True)
                        st.session_state.need_rerun = True
            
            elif answered and show_feedback:
                st.markdown("---")
                user_answer = st.session_state.answers.get(idx, "")
                
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
                
                st.markdown('<div class="option-grid">', unsafe_allow_html=True)
                
                for option_key, option_text in options.items():
                    option_upper = option_key.upper()
                    css_class = "option-card-interactive"
                    icon = option_key
                    
                    if answer_type == "single":
                        if option_upper == correct_answer:
                            css_class += " correct"
                            icon = "✅"
                            hint = "Correct Answer"
                        elif option_upper == user_answer.upper():
                            css_class += " wrong"
                            icon = "❌"
                            hint = "Your Selection (Incorrect)"
                        else:
                            hint = "Not Selected"
                    else:
                        user_answers_set = set([ans.strip().upper() for ans in user_answer.split(",")]) if user_answer else set()
                        correct_answers_set = set([ans.strip().upper() for ans in correct_answer.split(",")])
                        
                        if option_upper in correct_answers_set and option_upper in user_answers_set:
                            css_class += " correct"
                            icon = "✅"
                            hint = "Correct (You selected this)"
                        elif option_upper in correct_answers_set and option_upper not in user_answers_set:
                            css_class += " wrong"
                            icon = "⚠️"
                            hint = "Correct Answer (You missed this)"
                        elif option_upper not in correct_answers_set and option_upper in user_answers_set:
                            css_class += " wrong"
                            icon = "❌"
                            hint = "Incorrect (You selected this)"
                        else:
                            hint = "Not Selected"
                    
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
                
                if q.get('explanation'):
                    with st.expander("📖 Explanation", expanded=True):
                        st.info(q['explanation'])
                
                st.markdown("<div class='nav-btn-container'>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if idx > 0:
                        if st.button("← Previous Question", key=f"prev_feedback_{idx}", use_container_width=True, type="secondary"):
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
    
    # STATE 1: No Quiz - Setup
    else:
        st.subheader("📘 Upload Notes or Enter Topic for Quiz")
        
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
        
        if st.session_state.difficulty == "Difficult":
            st.markdown(
                '<div class="difficult-warning-box">'
                '⚠️ <strong>Difficult Mode:</strong> Questions may have single OR multiple correct answers! '
                'You won\'t know which type until submission. Full feedback shown after submission.'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        if st.session_state.quiz_source == "Notes":
            st.info("📤 Upload your study notes (PDF, DOCX, or TXT)")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "txt"],
                key="file_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                if uploaded_file != st.session_state.get("uploaded_file"):
                    st.session_state.notes_text = ""
                    st.session_state.uploaded_file = uploaded_file
                
                if not st.session_state.notes_text:
                    loader_placeholder = st.empty()
                    with loader_placeholder.container():
                        st.markdown(show_custom_loader(f"Extracting text from {uploaded_file.name}..."), unsafe_allow_html=True)
                    
                    suffix = os.path.splitext(uploaded_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    try:
                        st.session_state.notes_text = extract_text(temp_path)
                        loader_placeholder.empty()
                        
                        if st.session_state.notes_text:
                            st.success(f"✅ Successfully extracted {len(st.session_state.notes_text)} characters from {uploaded_file.name}")
                            
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
                            
                            try:
                                from Notes_Quiz_Section import store_notes_and_quizzes
                                store_notes_and_quizzes(
                                    user_id=st.session_state.user_id,
                                    notes_text=st.session_state.notes_text
                                )
                                st.info("📚 Notes saved to your knowledge base!")
                            except Exception as e:
                                st.warning(f"Note: Could not save to database: {str(e)}")
                                
                    except Exception as e:
                        loader_placeholder.empty()
                        st.error(f"❌ Error extracting text: {str(e)}")
                    finally:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                else:
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
                st.markdown("""
                <div style="text-align: center; padding: 40px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px; margin: 20px 0;">
                    <h3 style="color: #8a7bff;">📤 Upload Your Notes</h3>
                    <p>Drag and drop or click to upload PDF, DOCX, or TXT files</p>
                    <p style="font-size: 12px; color: #9aa3bf;">Supported formats: .pdf, .docx, .txt</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            if search_icon:
                search_icon_html = f'![search_icon-class](data:image/png;base64,{search_icon})'
            else:
                search_icon_html = "🔍 "
            st.info(f"{search_icon_html} Enter a  topic to generate a quiz ")
            
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
                    {"<br><small style='color: #ff6b6d;'>⚠️ Difficult mode: Question types hidden until submission</small>" if st.session_state.difficulty == "Difficult" else ""}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        generate_disabled = False
        generate_tooltip = ""
        
        if st.session_state.quiz_source == "Notes":
            if not st.session_state.notes_text:
                generate_disabled = True
                generate_tooltip = "Please upload and extract notes first"
        else:
            if not st.session_state.custom_topic or not st.session_state.custom_topic.strip():
                generate_disabled = True
                generate_tooltip = "Please enter a topic first"
        
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
            with quiz_loader_placeholder.container():
                loader_text = f"Generating {st.session_state.num_questions} {st.session_state.difficulty} questions..."
                st.markdown(show_custom_loader(loader_text), unsafe_allow_html=True)
            
            try:
                if st.session_state.quiz_source == "Notes":
                    input_text = st.session_state.notes_text
                    source_desc = "from your notes"
                else:
                    input_text = st.session_state.custom_topic
                    source_desc = f"about {st.session_state.custom_topic}"
                
                quiz_data = generate_quiz_from_notes(
                    notes_text=input_text,
                    user_id=st.session_state.user_id,
                    num_questions=st.session_state.num_questions,
                    difficulty=st.session_state.difficulty.lower(),
                )
                
                quiz_loader_placeholder.empty()
                
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
                quiz_loader_placeholder.empty()
                st.error(f"❌ Error generating quiz: {str(e)}")

# ---------- TAB 3: PROGRESS ----------
with tab3:
    st.session_state.active_tab = "Progress"
    
    st.subheader("📈 Your Learning Progress")
    
    if st.button("🔄 Refresh Progress", use_container_width=True, type="primary"):
        loader_placeholder = st.empty()
        with loader_placeholder.container():
            st.markdown(show_custom_loader("Refreshing progress..."), unsafe_allow_html=True)
        
        try:
            st.rerun()
        finally:
            loader_placeholder.empty()
    
    try:
        data = fetch_progress_from_pinecone(st.session_state.user_id)
        
        if data and data.get("progress"):
            summary = data.get("summary", {})
            progress_list = data.get("progress", [])
            
            if report_icon:
                report_icon_html = f'<img src="data:image/png;base64,{report_icon}" class="report-icon"> '
            else:
                report_icon_html = "📊 "
            
            st.markdown(f"### {report_icon_html} Overall Statistics", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col2:
                avg_score = summary.get('average_score', 0)
                scenario = "excellent" if avg_score >= 80 else "good" if avg_score >= 60 else "fair" if avg_score >= 40 else "needs_improvement"
                if target_icon:
                     target_icon_html = f'<img src="data:image/png;base64,{target_icon}"'
                else:
                    target_icon_html = "🎯 "
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{target_icon_html}</div>
                    <div class="metric-label">Average Score</div>
                    <div class="metric-value">{avg_score:.1f}</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Per Quiz</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(create_gradient_progress_bar(avg_score, "medium", scenario), unsafe_allow_html=True)
            
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
            
            with col4:
                topics = summary.get("topics_covered", [])
                topic_count = len(topics)
                topic_percentage = min(100, (topic_count / 10) * 100) if topics else 0
                scenario = "excellent" if topic_count >= 8 else "good" if topic_count >= 5 else "fair" if topic_count >= 3 else "needs_improvement"
                if book_icon:
                    book_icon_html = f'<img src="data:image/png;base64,{book_icon}" class="book-icon"> '
                else:
                    book_icon_html = "📖 "
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{book_icon_html}</div>
                    <div class="metric-label">Topics Covered</div>
                    <div class="metric-value">{topic_count}</div>
                    <div style="font-size: 12px; color: #9aa3bf;">Areas Studied</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(create_gradient_progress_bar(topic_percentage, "difficult", scenario), unsafe_allow_html=True)
            
            st.markdown("---")
            
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
            
            st.subheader("📋 Attempt History")
            
            for i, attempt in enumerate(progress_list[:10]):
                accuracy = attempt.get('accuracy', 0)
                score = attempt.get('score', 0)
                total = attempt.get('total', 1)
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
                        st.markdown("**Accuracy:**")
                        scenario = "excellent" if accuracy >= 80 else "good" if accuracy >= 60 else "fair" if accuracy >= 40 else "needs_improvement"
                        difficulty = attempt.get('difficulty', 'medium').lower()
                        st.markdown(create_gradient_progress_bar(accuracy, difficulty, scenario), unsafe_allow_html=True)
                        
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