# Chatbot.py - Chatbot Module with CAG Support
# FINAL VERSION — structured output, task-aware embeddings, locked top_k

import os
import json
import time
import re
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import pymupdf as fitz
import docx2txt
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient
from groq import Groq
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import wikipedia
import requests
import bcrypt
import jwt
import streamlit as st
from pydantic import BaseModel, Field

# ---------------- TIMEZONE SUPPORT ----------------
try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    print("[WARNING] pytz not installed. Install with: pip install pytz")
    TIMEZONE_AVAILABLE = False


# ---------------- TIMEZONE UTILITIES ----------------
def get_user_timezone(timezone_str=None):
    """Get user timezone"""
    if not TIMEZONE_AVAILABLE:
        return None
    try:
        if timezone_str:
            return pytz.timezone(timezone_str)
        return pytz.timezone(DEFAULT_TIMEZONE)
    except Exception:
        return pytz.timezone(DEFAULT_TIMEZONE)


def format_timestamp_to_local(timestamp, timezone_str=None, format_str="%Y-%m-%d %H:%M:%S"):
    """Format timestamp to local timezone"""
    if not TIMEZONE_AVAILABLE:
        return datetime.utcfromtimestamp(timestamp).strftime(format_str)
    try:
        dt_utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
        user_tz = get_user_timezone(timezone_str)
        return dt_utc.astimezone(user_tz).strftime(format_str)
    except Exception:
        return datetime.utcfromtimestamp(timestamp).strftime(format_str)


# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")

# --- LOCKED top_k, consistent with RAG_Eval.py ---
CHATBOT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# --- Structured output toggle (set to "false" to use streaming fallback) ---
USE_STRUCTURED_OUTPUT = (
    os.getenv("CHATBOT_STRUCTURED_OUTPUT", "true").lower() == "true"
)

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found!")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found!")

# ---------------- CONFIGURE GEMINI & PINECONE ----------------
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )
index = pc.Index(INDEX_NAME)


# ============================================
# STRUCTURED OUTPUT SCHEMA
# ============================================

class StudyBuddyResponse(BaseModel):
    """Structured response format for StudyBuddy."""

    summary: str = Field(
        description="A 1-2 sentence direct answer to the user's question"
    )

    key_points: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 6 bullet points with key takeaways. "
            "Leave empty for simple yes/no questions or single-fact answers."
        ),
        max_length=6,
    )

    details: Optional[str] = Field(
        default=None,
        description=(
            "Optional 1-2 paragraph elaboration. "
            "Only include if the summary alone is insufficient."
        ),
    )

    sources: List[str] = Field(
        default_factory=list,
        description=(
            "Which tools were used to answer this, "
            "e.g. ['notes'], ['wikipedia', 'web_search'], ['chat_history']"
        ),
    )

    follow_up: Optional[str] = Field(
        default=None,
        description="Optional suggested follow-up question",
    )


# ============================================
# USER ID VALIDATION
# ============================================

def validate_user_id(user_id: str) -> bool:
    """Validate user_id format - must start with 'user_'"""
    if not user_id:
        return False
    if user_id in ['default_user', 'Unknown', 'None', '']:
        return False
    if not user_id.startswith('user_'):
        return False
    return True


# ============================================
# TEXT FORMATTING UTILITIES
# ============================================

def clean_text(text: str) -> str:
    """Clean and format text - remove extra spaces, normalize line breaks"""
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' \n ', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    text = re.sub(r'([•\-*])\s+', r'\1 ', text)
    return text.strip()


def format_response(text: str) -> str:
    """
    Format response with proper line breaks.

    Preserves markdown headers, numbered lists, and bullet lists.
    Only collapses runs of 3+ newlines to 2.
    """
    if not text:
        return text

    text = text.replace('\r\n', '\n').replace('\r', '\n')

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*(\d+\.|[-*•]|\*\*|#{1,6}\s)', line.strip()):
            cleaned_lines.append(line.rstrip())
        else:
            cleaned_lines.append(re.sub(r'[ \t]+', ' ', line.rstrip()))

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def render_structured_response(response: StudyBuddyResponse) -> str:
    """Render the structured response as clean markdown for Streamlit."""
    parts = []

    # Summary (bold)
    if response.summary:
        parts.append(f"**{response.summary}**")

    # Key points (numbered list)
    if response.key_points:
        parts.append("")
        parts.append("**Key Points:**")
        for i, point in enumerate(response.key_points, 1):
            parts.append(f"{i}. {point}")

    # Details
    if response.details:
        parts.append("")
        parts.append("**Details:**")
        parts.append(response.details)

    # Sources
    if response.sources:
        parts.append("")
        parts.append(f"*Sources: {', '.join(response.sources)}*")

    # Follow-up
    if response.follow_up:
        parts.append("")
        parts.append(f"_{response.follow_up}_")

    return "\n".join(parts)


# ============================================
# DUPLICATE PREVENTION CACHE
# ============================================
_conversation_cache = {}
_conversation_cache_expiry = 60
_response_cache = {}
_response_cache_expiry = 30


def is_duplicate_response(response: str) -> bool:
    response_hash = hashlib.md5(response.encode()).hexdigest()
    if response_hash in _response_cache:
        return time.time() - _response_cache[response_hash] < _response_cache_expiry
    return False


def mark_response_sent(response: str):
    response_hash = hashlib.md5(response.encode()).hexdigest()
    _response_cache[response_hash] = time.time()


def is_duplicate_conversation(user_id: str, question: str, answer: str) -> bool:
    content_hash = hashlib.md5(f"{question}|{answer}".encode()).hexdigest()
    cache_key = f"{user_id}:{content_hash}"
    if cache_key in _conversation_cache:
        return time.time() - _conversation_cache[cache_key] < _conversation_cache_expiry
    return False


def mark_conversation_stored(user_id: str, question: str, answer: str):
    content_hash = hashlib.md5(f"{question}|{answer}".encode()).hexdigest()
    _conversation_cache[f"{user_id}:{content_hash}"] = time.time()


# ============================================
# GEMINI EMBEDDING — TASK-AWARE
# ============================================

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


def embed_text(text: str, task: str = "query") -> List[float]:
    """Embed text with Gemini Embedding 2, applying the correct task prefix."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    prefix = QUERY_PREFIX if task == "query" else DOC_PREFIX
    prefixed = (prefix + text.strip())[:10000]

    last_error = None
    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=[prefixed],
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            if result and result.embeddings and len(result.embeddings) > 0:
                emb = result.embeddings[0].values
                if emb is not None:
                    emb_list = list(emb)
                    if len(emb_list) == 768:
                        norm = sum(v * v for v in emb_list) ** 0.5
                        if norm > 0:
                            return [v / norm for v in emb_list]
                        return emb_list
            last_error = ValueError("Gemini returned empty or invalid embedding")
        except Exception as e:
            last_error = e
        if attempt < 2:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"embed_text failed after 3 attempts: {last_error}")


def embed_text_safe(text: str, task: str = "query") -> List[float]:
    """Safe wrapper. Returns None on failure (never random vectors)."""
    try:
        return embed_text(text, task=task)
    except RuntimeError as exc:
        print(f"[WARN] embed_text_safe failed: {exc}")
        return None


# ============================================
# PINECONE CRUD OPERATIONS
# ============================================

def generate_jwt(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except Exception:
        return None


def store_user(email: str, password: str) -> dict:
    """Store user in Pinecone with a deterministic ID."""
    existing = find_user_by_email(email)
    if existing:
        return None

    password_hash = bcrypt.hashpw(
        password.encode('utf-8'), bcrypt.gensalt()
    ).decode('utf-8')

    user_id = f"user_{int(time.time())}"

    text = f"user_auth:{email}"
    embedding = embed_text_safe(text, task="document")

    if embedding is None:
        print(f"[ERROR] Failed to embed user_auth for {email} — signup aborted")
        return None

    try:
        index.upsert(
            vectors=[{
                "id": f"{user_id}_auth",
                "values": embedding,
                "metadata": {
                    "user_id": user_id,
                    "type": "user_auth",
                    "email": email,
                    "password_hash": password_hash,
                    "created_at": datetime.now().isoformat()
                }
            }],
            namespace="users"
        )
    except Exception as e:
        print(f"[ERROR] Failed to upsert user: {e}")
        return None

    return {
        "user_id": user_id,
        "email": email
    }


def find_user_by_email(email: str) -> dict:
    """Find user by email using exact metadata filter (deterministic)."""
    try:
        results = index.query(
            vector=[0.0] * 768,
            top_k=1,
            include_metadata=True,
            namespace="users",
            filter={
                "type": {"$eq": "user_auth"},
                "email": {"$eq": email},
            }
        )

        for match in results.matches:
            if match.metadata and match.metadata.get("email") == email:
                return {
                    "user_id": match.metadata.get("user_id"),
                    "email": match.metadata.get("email"),
                    "password_hash": match.metadata.get("password_hash"),
                }

        return None

    except Exception as e:
        print(f"[ERROR] find_user_by_email: {e}")
        return None


def verify_user(email: str, password: str) -> dict:
    user = find_user_by_email(email)
    if not user:
        return None

    if bcrypt.checkpw(
        password.encode('utf-8'),
        user['password_hash'].encode('utf-8')
    ):
        return user
    return None


def store_conversation(user_id: str, question: str, answer: str, contexts: list):
    """Store conversation in Pinecone with retry + skip-on-failure."""
    if not validate_user_id(user_id):
        print(f"[WARNING] Invalid user_id: {user_id}, skipping save")
        return False

    if is_duplicate_conversation(user_id, question, answer):
        print(f"[INFO] Duplicate conversation for user {user_id}, skipping")
        return True

    question = clean_text(question)
    answer = clean_text(answer)

    text = f"Q: {question}\nA: {answer}"
    embedding = embed_text_safe(text, task="document")

    if embedding is None:
        print(f"[WARN] Failed to embed conversation — not stored")
        return False

    vector_id = f"chat_{user_id}_{int(time.time() * 1000)}"

    try:
        index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "user_id": user_id,
                    "type": "chat_history",
                    "question": question,
                    "answer": answer,
                    "contexts": json.dumps(contexts),
                    "timestamp": datetime.now().isoformat()
                }
            }],
            namespace="chat_history"
        )
        mark_conversation_stored(user_id, question, answer)
        print(f"[INFO] Stored conversation for user: {user_id}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to store conversation: {e}")
        return False


def get_user_history(user_id: str, limit: int = 50) -> list:
    """Fetch all chat history for a user (filter-only query)."""
    if not validate_user_id(user_id):
        return []

    try:
        results = index.query(
            vector=[0.0] * 768,
            top_k=limit,
            include_metadata=True,
            namespace="chat_history",
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": "chat_history"},
            }
        )
    except Exception as e:
        print(f"[ERROR] get_user_history: {e}")
        return []

    history = []
    seen_questions = set()

    for match in results.matches:
        if match.metadata:
            question = match.metadata.get('question', '')
            if question in seen_questions:
                continue
            seen_questions.add(question)

            history.append({
                "question": clean_text(question),
                "answer": clean_text(match.metadata.get('answer', '')),
                "contexts": json.loads(match.metadata.get('contexts', '[]')),
                "timestamp": match.metadata.get('timestamp', '')
            })

    return history


def get_conversation_context(user_id: str, query: str, top_k: int = 3) -> list:
    """Semantic search over chat history for CAG (uses query prefix)."""
    if not validate_user_id(user_id):
        return []

    query = clean_text(query)
    embedding = embed_text_safe(query, task="query")

    if embedding is None:
        return []

    try:
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="chat_history",
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": "chat_history"},
            }
        )
    except Exception as e:
        print(f"[ERROR] get_conversation_context: {e}")
        return []

    contexts = []
    seen_questions = set()

    for match in results.matches:
        if match.metadata:
            question = match.metadata.get('question', '')
            if question in seen_questions:
                continue
            seen_questions.add(question)

            contexts.append({
                "question": clean_text(question),
                "answer": clean_text(match.metadata.get('answer', '')),
                "score": match.score
            })

    return contexts


# ============================================
# GLOBAL CONTEXT STORAGE
# ============================================
_last_contexts = []


def get_last_contexts():
    return _last_contexts


def set_last_contexts(contexts):
    global _last_contexts
    _last_contexts = contexts


# ============================================
# TOP_K — LOCKED TO MATCH RAG_EVAL
# ============================================

def _count_user_notes(user_id: str) -> int:
    """Count total note chunks for a user (filter-only)."""
    if not validate_user_id(user_id):
        return 0

    try:
        results = index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=False,
            namespace="notes",
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": "notes"},
            },
        )
        return len(results.matches)
    except Exception:
        return 20


def calculate_top_k(total_chunks: int) -> int:
    """Locked production top_k — matches RAG_Eval.py."""
    return CHATBOT_TOP_K


# ============================================
# RAG CONTEXT RETRIEVAL
# ============================================

def retrieve_context(
    query: str,
    user_id: str = None,
    top_k: int = None,
    search_types: list = None,
    extract_answers: bool = False
) -> tuple:
    """Retrieve context from notes namespace (top_k locked)."""
    try:
        query = clean_text(query)
        q_emb = embed_text_safe(query, task="query")

        if q_emb is None:
            return "Failed to embed query.", []

        if top_k is None:
            top_k = CHATBOT_TOP_K

        answer_keywords = [
            'correct answer', 'answers', 'correct option', 'right answer',
            'what were the correct', 'correct choices'
        ]
        is_answer_query = (
            any(kw in query.lower() for kw in answer_keywords)
            or extract_answers
        )

        filter_dict = {}
        if user_id and validate_user_id(user_id):
            filter_dict["user_id"] = {"$eq": user_id}

        if search_types:
            filter_dict["type"] = {"$in": search_types}
        else:
            filter_dict["type"] = {"$eq": "notes"}

        resp = index.query(
            vector=q_emb,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            namespace="notes"
        )

        matches = getattr(resp, "matches", [])
        print(f"  [DEBUG] Found {len(matches)} matches for: {query[:30]}... (top_k={top_k})")

        if not matches:
            return "No relevant information found.", []

        contexts = []
        seen_texts = set()

        for match in matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            score = match.score or 0
            source_type = meta.get("type", meta.get("source", "unknown"))

            if not text and meta.get("type") == "quiz":
                quiz_data = meta.get("quiz_data", "")
                if quiz_data:
                    try:
                        quiz_json = json.loads(quiz_data)
                        if is_answer_query:
                            text = extract_answers_from_quiz(quiz_json)
                        else:
                            text = format_quiz_for_display(quiz_json, show_answers=True)
                    except Exception as e:
                        print(f"[WARNING] Error parsing quiz data: {e}")
                        text = "Quiz data available"

            if text and score > 0.1 and text not in seen_texts:
                seen_texts.add(text)
                text = clean_text(text)
                if len(text) > 3000:
                    text = text[:3000] + "..."
                contexts.append({
                    "text": text,
                    "source": source_type,
                    "score": score
                })

        contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_contexts = contexts[:3]

        if top_contexts:
            context_strings = []
            for c in top_contexts:
                text = clean_text(c['text'])
                context_strings.append(f"[From {c['source']}] {text}")
            result = "\n\n---\n\n".join(context_strings)
            return result, top_contexts

        return "No relevant information found.", []

    except Exception as e:
        print(f"[ERROR] retrieve_context: {e}")
        return "Sorry, couldn't retrieve information.", []


# ============================================
# CAG: SEARCH CHAT HISTORY TOOL
# ============================================

def search_chat_history_tool_func(query: str, user_id: str) -> str:
    try:
        if not validate_user_id(user_id):
            return "Invalid user ID."

        query = clean_text(query)
        chat_contexts = get_conversation_context(user_id, query, top_k=5)

        if not chat_contexts:
            return "No relevant previous conversations found."

        result = "💬 **Relevant conversations from your history:**\n\n"
        for i, chat in enumerate(chat_contexts, 1):
            question = clean_text(chat.get('question', ''))
            answer = clean_text(chat.get('answer', '')[:300])
            score = chat.get('score', 0)
            result += f"{i}. **Q:** {question}\n"
            result += f"   **A:** {answer}...\n"
            result += f"   (Relevance: {score:.2f})\n\n"

        return result
    except Exception as e:
        return f"Error searching chat history: {str(e)}"


# ============================================
# QUIZ FORMATTING HELPERS
# ============================================

def extract_answers_from_quiz(quiz_json: dict) -> str:
    try:
        questions = quiz_json.get('quiz', [])
        if not questions:
            return "No questions found in quiz."

        result = f"📝 **Correct Answers for Quiz:** {quiz_json.get('topic', 'Unknown')[:50]}...\n\n"

        for i, q in enumerate(questions, 1):
            question_text = clean_text(q.get('question', ''))
            answer = q.get('answer', 'N/A')
            answer_type = q.get('answer_type', 'single')

            if ',' in answer:
                answers = [a.strip() for a in answer.split(',')]
                answer_display = ', '.join(answers)
            else:
                answer_display = answer

            result += f"**Q{i}:** {question_text}\n"
            result += f"   ✅ **Answer:** {answer_display}\n"
            result += f"   📌 Type: {answer_type.upper()}\n\n"

        return result
    except Exception as e:
        return f"Error extracting answers: {str(e)}"


def format_quiz_for_display(quiz_json: dict, show_answers: bool = True) -> str:
    try:
        questions = quiz_json.get('quiz', [])
        if not questions:
            return "No questions found in quiz."

        result = f"📝 **Quiz Topic:** {quiz_json.get('topic', 'Unknown')[:80]}\n"
        result += f"📊 **Difficulty:** {quiz_json.get('difficulty', 'medium').upper()}\n\n"

        for i, q in enumerate(questions, 1):
            question_text = clean_text(q.get('question', ''))
            options = q.get('options', {})
            answer_type = q.get('answer_type', 'single')

            result += f"**Q{i}.** {question_text}\n"

            for opt_key, opt_text in options.items():
                result += f"   {opt_key}) {clean_text(opt_text)}\n"

            if answer_type == 'multiple':
                result += "   📌 [MULTIPLE CORRECT ANSWERS]\n"
            else:
                result += "   📌 [SINGLE CORRECT ANSWER]\n"

            if show_answers:
                answer = q.get('answer', 'N/A')
                if ',' in answer:
                    answers = [a.strip() for a in answer.split(',')]
                    answer_display = ', '.join(answers)
                else:
                    answer_display = answer
                result += f"   ✅ Correct: {answer_display}\n"

            result += "\n"

        return result
    except Exception as e:
        return f"Error formatting quiz: {str(e)}"


# ============================================
# TOOLS
# ============================================

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for encyclopedia articles about a topic."""
    try:
        query = clean_text(query)
        search_results = wikipedia.search(query)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"

        page = wikipedia.page(search_results[0])
        summary = clean_text(page.summary[:600])

        return f"📚 **Wikipedia: {page.title}**\n\n{summary}...\n\n🔗 {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = ', '.join(e.options[:5])
        return f"Multiple pages found. Options: {options}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily."""
    try:
        if not TAVILY_API_KEY:
            return "Tavily API key not configured."

        query = clean_text(query)
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5)

        if response and isinstance(response, dict):
            results = response.get('results', [])
            if results:
                formatted_results = []
                for i, result in enumerate(results[:3], 1):
                    title = clean_text(result.get('title', 'No title'))
                    content = clean_text(result.get('content', 'No content')[:300])
                    url = result.get('url', '')
                    formatted_results.append(f"{i}. **{title}**\n   {content}...\n   🔗 {url}")

                return "🌐 **Web Search Results:**\n\n" + "\n\n".join(formatted_results)

        return "No web search results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# ============================================
# MAIN CHATBOT FUNCTION
# ============================================

def get_gemini_response(
    user_input: str,
    history: list = None,
    user_id: str = None
) -> Generator[str, None, None]:
    """
    StudyBuddy with CAG support and structured output.

    When USE_STRUCTURED_OUTPUT=True: forces the agent to return a
    StudyBuddyResponse object, then renders it as clean markdown.

    When USE_STRUCTURED_OUTPUT=False: falls back to streaming raw text.
    """
    if not validate_user_id(user_id):
        yield "⚠️ Authentication error. Invalid user ID. Please login again."
        return

    user_input = clean_text(user_input)

    try:
        is_topic_query = (
            "list" in user_input.lower()
            and ("topic" in user_input.lower() or "studied" in user_input.lower())
        )

        relevant_contexts = []
        try:
            chat_contexts = get_conversation_context(user_id, user_input, top_k=3)
            if chat_contexts:
                print(f"[INFO] CAG: Found {len(chat_contexts)} relevant past conversations")
                relevant_contexts = chat_contexts
        except Exception as e:
            print(f"[WARNING] CAG retrieval failed: {e}")

        system_prompt = """You are StudyBuddy 🤖, a smart research assistant and tutor with memory.

Guidelines:
1. Use the available tools to look up information when needed
2. Always cite your sources when using tools
3. Be helpful, educational, and encouraging
4. Remember previous conversations and use them as context
5. When asked for a list of topics, summarize them concisely
6. Keep responses concise and focused on what was asked
7. NEVER repeat the same response twice

Available tools:
- search_notes: Search the user's personal notes and documents
- search_quizzes: Search the user's saved quizzes for relevant information
- search_chat_history: Search the user's previous conversations for context
- search_wikipedia: Search Wikipedia for encyclopedia knowledge
- web_search: Search the web for current news and real-time information

CRITICAL INSTRUCTIONS FOR STRUCTURED OUTPUT:
- Your "summary" field must directly answer the question in 1-2 sentences
- Your "key_points" field is for multi-part answers (lists, comparisons)
- For simple factual questions, leave "key_points" empty — just answer in summary
- The "details" field is only for elaboration the user actually needs
- The "sources" field must list which tools you used (e.g., ["notes"])
- The "follow_up" field is optional — only include if genuinely useful

WHEN THE USER ASKS "what is my notes about" or similar summary questions:
- Synthesize, do NOT paste raw content
- Extract key topics into "key_points"
- Keep the summary short and informative

WHEN THE USER ASKS "give me N points":
- Return EXACTLY N items in "key_points"
- Do not add extra or fewer

NEVER paste raw context or footnotes into your response.
"""

        if is_topic_query:
            system_prompt += """
CURRENT QUERY: User is asking for a list of topics they've studied.
- Search notes and quizzes for topics
- Extract unique topics from the context
- Present them as clean bullet points in "key_points"
- DO NOT show the full quiz or note content
"""

        if relevant_contexts:
            context_prompt = "\n\n💬 RECENT RELEVANT CONVERSATIONS:\n"
            for i, ctx in enumerate(relevant_contexts, 1):
                context_prompt += f"{i}. Q: {clean_text(ctx.get('question', ''))}\n"
                context_prompt += f"   A: {clean_text(ctx.get('answer', '')[:200])}...\n"
            system_prompt += context_prompt
            system_prompt += "\n\nUse this context to provide more coherent and continuous responses."

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        @tool
        def search_notes(query: str) -> str:
            """Search the user's personal notes and documents for information."""
            try:
                query = clean_text(query)
                context_str, contexts = retrieve_context(
                    query, user_id, top_k=None, search_types=["notes"]
                )
                set_last_contexts(contexts)

                if "list" in query.lower() and ("topic" in query.lower() or "studied" in query.lower()):
                    topics = []
                    for ctx in contexts:
                        if ctx.get('source') == 'notes':
                            text = ctx.get('text', '')
                            lines = text.split('\n')[:3]
                            for line in lines:
                                line = clean_text(line)
                                if line and len(line) < 80:
                                    topics.append(f"• {line}")

                    if topics:
                        return "📚 **Topics from your notes:**\n\n" + '\n'.join(topics[:5])
                    return "No specific topics found in your notes."

                if context_str and "No relevant" not in context_str:
                    return f"📄 **From your notes:**\n\n{context_str}"
                return "No relevant information found in your notes."
            except Exception as e:
                return f"Error searching notes: {str(e)}"

        @tool
        def search_quizzes(query: str) -> str:
            """Search the user's saved quizzes for relevant information."""
            try:
                query = clean_text(query)
                context_str, contexts = retrieve_context(
                    query, user_id, top_k=None, search_types=["quiz"]
                )
                set_last_contexts(contexts)

                if "list" in query.lower() and ("topic" in query.lower() or "studied" in query.lower()):
                    topics = []
                    for ctx in contexts:
                        if ctx.get('source') == 'quiz':
                            quiz_data = ctx.get('full_data', {})
                            if quiz_data:
                                topic = quiz_data.get('topic', 'Unknown')
                                difficulty = quiz_data.get('difficulty', 'medium')
                                num_questions = len(quiz_data.get('quiz', []))
                                topic_clean = clean_text(topic)[:60]
                                topics.append(
                                    f"• {topic_clean}\n  Difficulty: {difficulty.upper()}, {num_questions} questions"
                                )

                    if topics:
                        return "📚 **Topics from your quizzes:**\n\n" + '\n\n'.join(topics[:5])
                    return "No specific quiz topics found."

                if context_str and "No relevant" not in context_str:
                    return f"📚 **From your quizzes:**\n\n{context_str}"
                return "No relevant quizzes found."
            except Exception as e:
                return f"Error searching quizzes: {str(e)}"

        @tool
        def search_chat_history_wrapper(query: str) -> str:
            """Search the user's previous conversations for relevant information."""
            try:
                return search_chat_history_tool_func(query, user_id)
            except Exception as e:
                return f"Error searching chat history: {str(e)}"

        tools = [
            search_notes,
            search_quizzes,
            search_chat_history_wrapper,
            search_wikipedia,
            web_search,
        ]

        # ---- Build the agent ----
        if USE_STRUCTURED_OUTPUT:
            agent = create_agent(
                model=llm,
                tools=tools,
                checkpointer=MemorySaver(),
                system_prompt=system_prompt,
                response_format=StudyBuddyResponse,
            )
        else:
            agent = create_agent(
                model=llm,
                tools=tools,
                checkpointer=MemorySaver(),
                system_prompt=system_prompt,
            )

        # ---- Build message history ----
        messages = []
        if relevant_contexts:
            for ctx in relevant_contexts[:2]:
                messages.append(("user", clean_text(ctx.get('question', ''))))
                messages.append(("assistant", clean_text(ctx.get('answer', '')[:500])))

        if history:
            for h in history[-5:]:
                if h.get('role') == 'user':
                    messages.append(("user", clean_text(h.get('message', ''))))
                else:
                    messages.append(("assistant", clean_text(h.get('message', ''))))

        messages.append(("user", user_input))

        input_state = {"messages": messages}
        config = {
            "configurable": {
                "thread_id": user_id or "default_thread"
            }
        }

        # ============================================
        # STRUCTURED OUTPUT PATH
        # ============================================
        if USE_STRUCTURED_OUTPUT:
            try:
                final_state = None
                for chunk in agent.stream(
                    input_state,
                    stream_mode="values",
                    config=config,
                ):
                    final_state = chunk
                    last_msg = chunk["messages"][-1]

                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"\n🔧 Using tools:")
                        for tool_call in last_msg.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            print(f"   - {tool_name}")

                structured = None
                if final_state:
                    structured = final_state.get("structured_response")

                if structured and isinstance(structured, StudyBuddyResponse):
                    rendered = render_structured_response(structured)

                    # Save conversation (use summary as answer)
                    try:
                        store_conversation(
                            user_id=user_id,
                            question=user_input,
                            answer=structured.summary or rendered[:500],
                            contexts=get_last_contexts()[:3],
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to store conversation: {e}")

                    yield rendered
                    return
                else:
                    # Structured output didn't come through — fall back
                    print("[WARN] structured_response missing, falling back to streaming")
                    if final_state and final_state.get("messages"):
                        last_content = final_state["messages"][-1].content or ""
                        yield format_response(last_content)
                        return

            except Exception as structured_error:
                print(f"[WARN] Structured output failed: {structured_error}")
                print("[INFO] Falling back to streaming path...")
                # Fall through to the streaming path below

        # ============================================
        # STREAMING FALLBACK PATH
        # ============================================
        collected_response = ""
        seen_responses = set()

        try:
            for chunk in agent.stream(input_state, stream_mode="values", config=config):
                last_msg = chunk["messages"][-1]

                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"\n🔧 Using tools:")
                    for tool_call in last_msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        print(f"   - {tool_name}: {tool_args.get('query', '')}")

                if hasattr(last_msg, 'content') and last_msg.content:
                    content = last_msg.content
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    if content_hash not in seen_responses:
                        seen_responses.add(content_hash)
                        collected_response = content

            # Apply formatting once at the end
            final_response = format_response(collected_response)
            yield final_response

        except Exception as stream_error:
            print(f"[WARNING] Agent streaming failed: {stream_error}")
            print("[INFO] Falling back to direct response with CAG context...")

            note_context_str, note_contexts = retrieve_context(
                user_input, user_id, top_k=None, search_types=["notes"]
            )
            quiz_context_str, quiz_contexts = retrieve_context(
                user_input, user_id, top_k=None, search_types=["quiz"]
            )
            chat_contexts = get_conversation_context(user_id, user_input, top_k=3)

            combined_context = ""
            if note_context_str and "No relevant" not in note_context_str:
                combined_context += f"Notes: {note_context_str}\n\n"
            if quiz_context_str and "No relevant" not in quiz_context_str:
                combined_context += f"Quizzes: {quiz_context_str}\n\n"

            chat_context_str = ""
            if chat_contexts:
                chat_context_str = "Previous relevant conversations:\n"
                for i, ctx in enumerate(chat_contexts, 1):
                    chat_context_str += f"{i}. Q: {clean_text(ctx.get('question', ''))}\n"
                    chat_context_str += f"   A: {clean_text(ctx.get('answer', '')[:200])}...\n"
                combined_context += chat_context_str

            set_last_contexts(note_contexts + quiz_contexts + chat_contexts)

            fallback_llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )

            fallback_prompt = f"""You are StudyBuddy, a helpful tutor with memory.

Context from notes and quizzes:
{combined_context if combined_context else "No context available"}

User question: {user_input}

Provide a helpful response using the context above.
- If the question asks for correct answers, extract ONLY the answers
- If the question asks for questions, show the questions
- If the question asks for the full quiz, show everything
- DO NOT repeat the entire context unless specifically asked
- Be concise and directly answer the question
- Format responses with clean spacing"""

            response = fallback_llm.invoke(fallback_prompt)
            collected_response = format_response(response.content)

            yield collected_response

    except Exception as e:
        print(f"[ERROR] get_gemini_response: {e}")
        import traceback
        traceback.print_exc()
        yield f"I'm having trouble processing your request. Error: {str(e)}"


def get_studybuddy_response(user_input: str, history: list = None, user_id: str = None):
    """Alias for get_gemini_response"""
    return get_gemini_response(user_input, history, user_id)