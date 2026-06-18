import os
import json
import time
import re
import random
from datetime import datetime
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
import wikipedia
# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found!")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found!")


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
# ---------------- TIMEZONE SUPPORT ----------------
try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    print("[WARNING] pytz not installed. Install with: pip install pytz")
    TIMEZONE_AVAILABLE = False

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
# ---------------- FETCH USER PROGRESS ----------------
def fetch_progress_from_pinecone(user_id: str, user_timezone=None):
    """Fetch and summarize user progress with timezone support"""
    try:
        query_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = index.query(
            vector=query_vector,
            filter={"user_id": {"$eq": user_id}, "type": {"$eq": "progress"}}, 
            top_k=50, 
            include_metadata=True
        )
        
        progress_list = []
        for match in getattr(results, "matches", []):
            meta = match.metadata or {}
            try:
                data = json.loads(meta.get("progress_data", "{}"))
                if data and isinstance(data, dict):
                    timestamp = data.get('timestamp', 0)
                    if timestamp:
                        stored_timezone = meta.get('user_timezone', user_timezone)
                        data['local_time'] = format_timestamp_to_local(timestamp, stored_timezone)
                        data['date'] = format_timestamp_to_local(timestamp, stored_timezone, "%Y-%m-%d")
                        data['relative_time'] = get_relative_time(timestamp, stored_timezone)
                    progress_list.append(data)
            except Exception as e:
                print(f"Error parsing progress data: {e}")
                continue
        
        progress_list.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        if not progress_list:
            return {
                "user_id": user_id, 
                "progress": [], 
                "summary": {
                    "total_attempts": 0, 
                    "average_score": 0, 
                    "average_accuracy": 0,
                    "topics_covered": []
                }
            }
        
        total_attempts = len(progress_list)
        average_score = sum(p.get("score", 0) for p in progress_list) / total_attempts
        average_accuracy = sum(p.get("accuracy", 0) for p in progress_list) / total_attempts
        
        topics = set()
        for p in progress_list:
            topic = p.get("topic", "")
            if topic:
                topics.add(topic)
        
        one_week_ago = time.time() - (7 * 24 * 3600)
        recent_activity = [p for p in progress_list if p.get('timestamp', 0) > one_week_ago]
        
        summary = {
            "total_attempts": total_attempts,
            "average_score": round(average_score, 2),
            "average_accuracy": round(average_accuracy, 2),
            "topics_covered": list(topics)[:10],
            "recent_activity": len(recent_activity)
        }
        
        return {"user_id": user_id, "progress": progress_list, "summary": summary}
        
    except Exception as e:
        print(f"[ERROR] fetch_progress_from_pinecone: {e}")
        return {"user_id": user_id, "progress": [], "summary": {}}