# pages/main.py - Main Application Page
import streamlit as st
import os
import tempfile
import base64
import time
import streamlit.components.v1 as components

# ============================================
# IMPORT AUTHENTICATION MODULE
# ============================================
from auth import init_session_state, require_auth

# ============================================
# REQUIRE AUTHENTICATION WITH TIME GAP
# ============================================

# Add small delay to handle async issues
time.sleep(0.5)

# This will check both session state and query params
require_auth()

# Add another small delay after authentication
time.sleep(0.5)


# Now proceed with the rest of the imports
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

# ============================================
# LOAD CSS FROM EXTERNAL FILE
# ============================================
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

# ============================================
# Base64 Image/GIF Loader
# ============================================
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

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="AI Study Assistant — StudyBuddy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Session State Setup
# ============================================
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

# ============================================
# CONVERSATION SAVING WITH USER ID
# ============================================
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

# ============================================
# Chat Helpers
# ============================================
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

# ============================================
# Header Section
# ============================================
# Get user info from either st.user or session state
user_email = st.user.email if hasattr(st, 'user') and st.user else st.session_state.get('user_email', 'Unknown')
user_id = st.user.user_id if hasattr(st, 'user') and st.user else st.session_state.get('user_id', 'Unknown')

st.markdown(
    f"""
    <div class="header-card">
        <div class="header-inner">
            <div class="header-avatar">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
            </div>
            <div>
                <h1>Meet StudyBuddy</h1>
                <p>Your Personal AI Learning Assistant with Advanced Quiz Features</p>
                <p style="font-size:12px; color:#8a7bff;">
                    👤 Logged in as: {user_email}
                </p>
                <p style="font-size:11px; color:#64748b;">
                    🆔 User ID: {user_id}
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================
# SIDEBAR: Chat History & Logout
# ============================================
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
    st.markdown(f"**User ID:** {user_id}")
    
    # Custom Logout button (since st.logout doesn't accept callback in this version)
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        # Clear session state
        for key in ['token', 'user_id', 'user_email', 'logged_in', 'auth_checked']:
            if key in st.session_state:
                st.session_state[key] = None if key != 'logged_in' else False
        
        # Clear localStorage via JavaScript
        components.html("""
        <script>
            localStorage.removeItem('auth_token');
            localStorage.removeItem('token_timestamp');
            window.location.href = window.location.pathname;
        </script>
        """, height=0)
        
        # Redirect to login
        st.switch_page("app.py")
        st.stop()
    
    st.markdown("---")
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



# ============================================
# MAIN TABS WITH ICONS
# ============================================
if tab_chat_icon:
    chat_icon = f"![Icon](data:image/png;base64,{tab_chat_icon}) "
else:
    chat_icon = "💬"
if notes_icon:
    notes_icon_display = f"![Icon](data:image/png;base64,{notes_icon}) "
else:
    notes_icon_display = "📝"
if progress_icon:
    progress_icon_display = f"![Icon](data:image/png;base64,{progress_icon}) "
else:
    progress_icon_display = "📊"

tab1, tab2, tab3 = st.tabs([
    f"{chat_icon} Chat",
    f"{notes_icon_display} Notes & Quiz", 
    f"{progress_icon_display} Progress"
])

# ============================================
# TAB 1: CHAT
# ============================================
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
        
        # Handle streaming response
        if st.session_state.get("ai_responding", False) and st.session_state.get("last_user_message"):
            try:
                response_stream = get_gemini_response(
                    st.session_state.last_user_message, 
                    st.session_state.last_chat_messages, 
                    st.user.user_id if st.user else None
                )
                
                user_question = st.session_state.last_user_message
                
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
                    
                    if full_response and user_question:
                        add_message_to_current_session("bot", full_response)
                        save_conversation_to_pinecone(
                            user_id=st.user.user_id if st.user else None,
                            question=user_question,
                            answer=full_response,
                            contexts=[]
                        )
                    
                    st.session_state._full_response = full_response
                
                st.write_stream(response_generator)
                scroll_to_bottom()
                
                st.session_state.ai_responding = False
                st.session_state.streaming_message = ""
                st.session_state.last_user_message = None
                st.session_state.last_chat_messages = None
                st.session_state._full_response = ""
                
                st.rerun()
                
            except Exception as e:
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

# ============================================
# TAB 2: NOTES & QUIZ
# ============================================
with tab2:
    st.session_state.active_tab = "Notes & Quiz"
    
    # Quiz Completed State
    if st.session_state.completed and st.session_state.quiz:
        st.markdown("---")
        st.subheader("🎉 Quiz Results")
        
        try:
            answers_for_evaluation = {str(k): v for k, v in st.session_state.answers.items()}
            
            result = evaluate_quiz_attempt(
                st.session_state.quiz,
                answers_for_evaluation,
                st.user.user_id if st.user else None
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
            
            # Detailed feedback
            with st.expander("📝 Review All Questions", expanded=False):
                for i, feedback in enumerate(result.get("feedback", [])):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {feedback.get('question', '')}")
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
                            st.write(f"**Your Answer:** {', '.join(user_answers) if user_answers else 'None'}")
                            st.write(f"**Correct Answer:** {', '.join(correct_answers)}")
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
                    st.rerun()
            
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
    
    # Quiz Active State
    elif st.session_state.quiz and not st.session_state.completed:
        st.info("Quiz is in progress...")
    
    # No Quiz - Setup State
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
                ["Easy", "Medium", "Hard", "Difficult"]
            )
        with col3:
            st.session_state.num_questions = st.selectbox("Questions:", [5, 10, 15])
        
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
        
        else:
            st.info("🔍 Enter a topic to generate a quiz")
            st.session_state.custom_topic = st.text_input(
                "Enter your topic:",
                placeholder="e.g., Machine Learning, World History, Biology, Python Programming, etc.",
                value=st.session_state.custom_topic,
                key="topic_input"
            )
        
        st.markdown("---")
        
        generate_disabled = False
        if st.session_state.quiz_source == "Notes":
            if not st.session_state.notes_text:
                generate_disabled = True
        else:
            if not st.session_state.custom_topic or not st.session_state.custom_topic.strip():
                generate_disabled = True
        
        quiz_loader_placeholder = st.empty()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_clicked = st.button(
                "🚀 Generate Quiz",
                disabled=generate_disabled,
                use_container_width=True,
                type="primary"
            )
        
        if generate_clicked:
            with quiz_loader_placeholder.container():
                loader_text = f"Generating {st.session_state.num_questions} {st.session_state.difficulty} questions..."
                st.markdown(show_custom_loader(loader_text), unsafe_allow_html=True)
            
            try:
                if st.session_state.quiz_source == "Notes":
                    input_text = st.session_state.notes_text
                else:
                    input_text = st.session_state.custom_topic
                
                quiz_data = generate_quiz_from_notes(
                    notes_text=input_text,
                    user_id=st.user.user_id if st.user else None,
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
                    
                    st.success(f"✅ Successfully generated {len(quiz_data['quiz'])} questions!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to generate quiz. Please try again.")
                    
            except Exception as e:
                quiz_loader_placeholder.empty()
                st.error(f"❌ Error generating quiz: {str(e)}")

# ============================================
# TAB 3: PROGRESS
# ============================================
with tab3:
    st.session_state.active_tab = "Progress"
    
    st.subheader("📈 Your Learning Progress")
    
    if st.button("🔄 Refresh Progress", use_container_width=True, type="primary"):
        st.rerun()
    
    try:
        data = fetch_progress_from_pinecone(st.user.user_id if st.user else None)
        
        if data and data.get("progress"):
            summary = data.get("summary", {})
            progress_list = data.get("progress", [])
            
            st.markdown("### 📊 Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_attempts = summary.get("total_attempts", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📚</div>
                    <div class="metric-label">Total Attempts</div>
                    <div class="metric-value">{total_attempts}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_score = summary.get('average_score', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-label">Average Score</div>
                    <div class="metric-value">{avg_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_acc = summary.get('average_accuracy', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-label">Average Accuracy</div>
                    <div class="metric-value">{avg_acc:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                topics = summary.get("topics_covered", [])
                topic_count = len(topics)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📖</div>
                    <div class="metric-label">Topics Covered</div>
                    <div class="metric-value">{topic_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
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
                
                with st.expander(f"📅 {time.strftime('%Y-%m-%d %H:%M', time.localtime(attempt.get('timestamp', time.time())))} - Score: {score}/{total} - Accuracy: {accuracy}%", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Topic:** {attempt.get('topic', 'Unknown Topic')}")
                        st.write(f"**Difficulty:** {attempt.get('difficulty', 'Medium').title()}")
                        st.write(f"**Source:** {attempt.get('source', 'Unknown')}")
        
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

# ============================================
# Rerun logic at the end
# ============================================
if st.session_state.get("need_rerun", False):
    st.session_state.need_rerun = False
    st.rerun()