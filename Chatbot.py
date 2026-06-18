import os
import json
import time
import re
import random
from datetime import datetime
from turtle import st
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import fitz  # PyMuPDF
import docx2txt  # Word extraction
import langgraph  # For agentic capabilities and tool usage
import langchain  # For potential future use in chaining LLM calls and tools
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from tavily import TavilyClient
import wikipedia
# ---------------- TIMEZONE SUPPORT ----------------
try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    print("[WARNING] pytz not installed. Install with: pip install pytz")
    TIMEZONE_AVAILABLE = False
# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found!")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found!")
if not SERPAPI_KEY:
    raise ValueError("❌ SERPAPI_KEY not found!")

# ---------------- CONFIGURE GEMINI & PINECONE ----------------
# Initialize Gemini client with latest google-genai SDK
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to Pinecone index
existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )
index = pc.Index(INDEX_NAME)
# ---------------- TIMEZONE UTILITIES ----------------
def get_user_timezone(timezone_str=None):
    """Get timezone object from string, fallback to default if invalid"""
    if not TIMEZONE_AVAILABLE:
        return None
    
    try:
        if timezone_str:
            return pytz.timezone(timezone_str)
        else:
            return pytz.timezone(DEFAULT_TIMEZONE)
    except pytz.UnknownTimeZoneError:
        print(f"[WARNING] Unknown timezone: {timezone_str}, using {DEFAULT_TIMEZONE}")
        return pytz.timezone(DEFAULT_TIMEZONE)

def format_timestamp_to_local(timestamp, timezone_str=None, format_str="%Y-%m-%d %H:%M:%S"):
    """Convert UTC timestamp to local timezone string"""
    if not TIMEZONE_AVAILABLE:
        dt = datetime.utcfromtimestamp(timestamp)
        return dt.strftime(format_str)
    
    try:
        dt_utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
        user_tz = get_user_timezone(timezone_str)
        dt_local = dt_utc.astimezone(user_tz)
        return dt_local.strftime(format_str)
    except Exception as e:
        print(f"[ERROR] format_timestamp_to_local: {e}")
        dt = datetime.utcfromtimestamp(timestamp)
        return dt.strftime(format_str)

def get_relative_time(timestamp, timezone_str=None):
    """Get human-readable relative time (e.g., '2 hours ago')"""
    now = time.time()
    diff = now - timestamp
    
    if diff < 60:
        return "just now"
    elif diff < 3600:
        minutes = int(diff / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < 604800:
        days = int(diff / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return format_timestamp_to_local(timestamp, timezone_str, "%b %d, %Y")

def get_common_timezones():
    """Return list of common timezones for dropdown selection"""
    if not TIMEZONE_AVAILABLE:
        return ["UTC"]
    
    return [
        'UTC',
        'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
        'America/Toronto', 'America/Vancouver', 'Europe/London', 'Europe/Paris',
        'Europe/Berlin', 'Europe/Moscow', 'Asia/Kolkata', 'Asia/Singapore',
        'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Dubai', 'Australia/Sydney',
        'Australia/Perth', 'Pacific/Auckland', 'Africa/Johannesburg',
        'America/Sao_Paulo', 'America/Mexico_City'
    ]

def get_current_time_in_timezone(timezone_str=None):
    """Get current time formatted in user's timezone"""
    current_time = time.time()
    return format_timestamp_to_local(current_time, timezone_str)

def format_timestamp_for_display(timestamp, user_timezone=None):
    """Format timestamp for nice display in UI"""
    if not timestamp:
        return "N/A"
    local_time = format_timestamp_to_local(timestamp, user_timezone)
    relative_time = get_relative_time(timestamp, user_timezone)
    return f"{local_time} ({relative_time})"

# ---------------- RAG CONTEXT RETRIEVAL ----------------
def retrieve_context(query: str, user_id: str = None, top_k: int = 5) -> str:
    """Enhanced RAG with filtering and relevance scoring"""
    try:
        q_emb = embed_text(query)
        
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = {"$eq": user_id}
        
        if filter_dict:
            resp = index.query(
                vector=q_emb, 
                filter=filter_dict,
                top_k=top_k, 
                include_metadata=True,
                include_values=False
            )
        else:
            resp = index.query(
                vector=q_emb, 
                top_k=top_k, 
                include_metadata=True,
                include_values=False
            )
        
        contexts = []
        seen_texts = set()
        
        for match in getattr(resp, "matches", []):
            meta = match.metadata or {}
            text = meta.get("text", "")
            score = match.score or 0
            
            if text and score > 0.3 and text not in seen_texts:
                seen_texts.add(text)
                source_type = meta.get("type", "unknown")
                context_text = f"[From {source_type}]: {text}"
                
                if len(contexts) < 3:
                    contexts.append(context_text)
        
        if contexts:
            return "\n\n---\n\n".join(contexts)
        return ""
        
    except Exception as e:
        print(f"[ERROR] retrieve_context: {e}")
        return ""
# ---------------- CHAT HISTORY FUNCTIONS ----------------
def save_chat_history(user_id: str, chat_history: list, user_timezone=None):
    """Save chat history to Pinecone"""
    try:
        if not chat_history or not isinstance(chat_history, list):
            return False
            
        chat_json = json.dumps(chat_history[-50:])
        emb = embed_text(chat_json)
        
        if all(abs(v) < 0.0001 for v in emb):
            emb = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        metadata = {
            "type": "chat", 
            "chat_data": chat_json, 
            "user_id": user_id,
            "timestamp": time.time(),
            "source": "chat_history"
        }
        
        if user_timezone and TIMEZONE_AVAILABLE:
            metadata.update({
                "local_time": format_timestamp_to_local(time.time(), user_timezone),
                "user_timezone": user_timezone
            })
        
        vectors = [{
            "id": f"{user_id}_chat_{int(time.time())}",
            "values": emb,
            "metadata": metadata
        }]
        
        index.upsert(vectors=vectors)
        print(f"[INFO] Chat history saved for user {user_id}")
        return True
    except Exception as e:
        print(f"[ERROR] save_chat_history: {e}")
        return False
#----------------- FETCH CHAT HISTORY ----------------
def fetch_chat_history(user_id: str, user_timezone=None):
    """Fetch complete chat history for a user"""
    try:
        query_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = index.query(
            vector=query_vector,
            filter={"user_id": {"$eq": user_id}, "type": {"$eq": "chat"}}, 
            top_k=10,
            include_metadata=True
        )
        
        chat_history = []
        for match in getattr(results, "matches", []):
            meta = match.metadata or {}
            try:
                data = json.loads(meta.get("chat_data", "[]"))
                if isinstance(data, list):
                    for chat in data:
                        if isinstance(chat, dict) and 'timestamp' in chat:
                            chat['local_time'] = format_timestamp_to_local(
                                chat['timestamp'], 
                                user_timezone or meta.get('user_timezone')
                            )
                    chat_history.extend(data)
            except Exception as e:
                print(f"Error parsing chat data: {e}")
                continue
        
        unique_chats = {}
        for chat in chat_history:
            if isinstance(chat, dict):
                key = f"{chat.get('role','')}_{chat.get('message','')}_{chat.get('timestamp',0)}"
                unique_chats[key] = chat
        
        sorted_chats = sorted(unique_chats.values(), key=lambda x: x.get("timestamp", 0))
        return sorted_chats[-100:]
        
    except Exception as e:
        print(f"[ERROR] fetch_chat_history: {e}")
        return []
#---------------- EMBEDDING FUNCTION ----------------
def embed_text(text: str):
    """Generate embeddings using latest google-genai SDK with gemini-embedding-2 and output_dimensionality=768"""
    try:
        if not text or not text.strip():
            print("[WARNING] Empty text provided for embedding")
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        text = text.strip()
        if len(text) > 10000:
            text = text[:10000]
        
        print(f"[DEBUG] Embedding text length: {len(text)}")
        
        # CORRECT METHOD using google-genai SDK with gemini-embedding-2 and output_dimensionality
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[text],  # contents must be a list
            config=types.EmbedContentConfig(
                output_dimensionality=768  # Forces 768 dimensions for Pinecone compatibility
            )
        )
        
        # Extract embedding from response
        emb = None
        if result and result.embeddings and len(result.embeddings) > 0:
            if hasattr(result.embeddings[0], 'values'):
                emb = result.embeddings[0].values
            elif isinstance(result.embeddings[0], list):
                emb = result.embeddings[0]
            else:
                emb = list(result.embeddings[0])
        
        if emb is None:
            print(f"[ERROR] Could not extract embedding from response: {type(result)}")
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        # With output_dimensionality=768, we should get exactly 768 dimensions
        if len(emb) != 768:
            print(f"[WARNING] Embedding length {len(emb)} != 768, adjusting")
            if len(emb) < 768:
                emb = emb + [random.uniform(0.001, 0.002) for _ in range(768 - len(emb))]
            else:
                emb = emb[:768]
        
        # Add small noise if all zeros
        if all(abs(v) < 0.0001 for v in emb):
            print(f"[WARNING] Embedding contains mostly zeros, adding small noise")
            emb = [v + (random.random() * 0.0001 - 0.00005) for v in emb]
        
        print(f"[INFO] Generated embedding of length {len(emb)}")
        return emb
        
    except Exception as e:
        print(f"[ERROR] embed_text: {e}")
        import traceback
        traceback.print_exc()
        return [random.uniform(0.01, 0.02) for _ in range(768)]


# ---------------- ENHANCED CHATBOT WITH RAG (USING LATEST SDK) ----------------
def get_gemini_response(user_input: str, history: list = None, user_id: str = None) -> Generator[str, None, None]:
    """Chatbot with proper streaming response using latest google-genai SDK"""
    try:
        casual_phrases = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]
        is_casual = any(p in user_input.lower() for p in casual_phrases)
        
        history_text = ""
        if history:
            recent_history = history[-10:]
            history_text = "\n".join([
                f"{h.get('role', 'user').capitalize()}: {h.get('message', '')}"
                for h in recent_history
            ])
        
        context = ""
        if user_id and not is_casual:
            context = retrieve_context(user_input, user_id)
        
        if context:
            prompt = f"""You are StudyBuddy 🤖, an AI tutor. Use the following context from the user's notes to provide accurate, helpful answers.

RELEVANT STUDY CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUERY: {user_input}

Instructions:
1. If the context contains relevant information, use it to answer accurately
2. If the context doesn't help, use your general knowledge
3. Keep answers concise, educational, and encouraging
4. Ask follow-up questions to deepen understanding when appropriate

AI Tutor:"""
        else:
            prompt = f"""You are StudyBuddy 🤖, a friendly AI tutor. 

CHAT HISTORY:
{history_text}

USER QUERY: {user_input}

Provide a helpful, engaging response. If asking about study topics, offer to help with notes or quizzes.

AI Tutor:"""
        
        # Use gemini-2.5-flash for best performance
        model = "gemini-2.5-flash"
        
        # Generate streaming response using latest SDK
        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
        
        # Yield chunks for proper streaming
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
            elif hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                yield part.text
                
    except Exception as e:
        print(f"[ERROR] get_gemini_response: {e}")
        def error_generator():
            yield "I encountered an error. Please try again or rephrase your question."
        return error_generator()