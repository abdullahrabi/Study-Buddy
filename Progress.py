import os
import json
import time
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found!")

# ---------------- CONFIGURE PINECONE ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    raise ValueError(f"❌ Index '{INDEX_NAME}' does not exist. Please run notes_quiz.py first.")

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
    if not TIMEZONE_AVAILABLE:
        return None
    try:
        if timezone_str:
            return pytz.timezone(timezone_str)
        return pytz.timezone(DEFAULT_TIMEZONE)
    except:
        return pytz.timezone(DEFAULT_TIMEZONE)

def format_timestamp_to_local(timestamp, timezone_str=None, format_str="%Y-%m-%d %H:%M:%S"):
    if not TIMEZONE_AVAILABLE:
        return datetime.utcfromtimestamp(timestamp).strftime(format_str)
    try:
        dt_utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
        user_tz = get_user_timezone(timezone_str)
        return dt_utc.astimezone(user_tz).strftime(format_str)
    except:
        return datetime.utcfromtimestamp(timestamp).strftime(format_str)

def get_relative_time(timestamp, timezone_str=None):
    now = time.time()
    diff = now - timestamp
    if diff < 60:
        return "just now"
    elif diff < 3600:
        return f"{int(diff / 60)} minutes ago"
    elif diff < 86400:
        return f"{int(diff / 3600)} hours ago"
    elif diff < 604800:
        return f"{int(diff / 86400)} days ago"
    return format_timestamp_to_local(timestamp, timezone_str, "%b %d, %Y")

# ---------------- EMBEDDING FUNCTION ----------------
def embed_text(text: str):
    """Simple embedding function for Pinecone queries."""
    try:
        if not text or not text.strip():
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        # For progress, we just need a consistent vector for filtering
        # Use a hash-based deterministic vector
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash to vector
        vector = []
        for i in range(768):
            byte_val = hash_bytes[i % len(hash_bytes)]
            vector.append((byte_val / 255.0) * 0.02 + 0.01)
        
        return vector
    except:
        return [random.uniform(0.01, 0.02) for _ in range(768)]

# ---------------- DUPLICATE PREVENTION CACHE ----------------
_progress_cache = {}
_cache_expiry = 300  # 5 minutes

def get_cache_key(user_id: str, hash_value: str) -> str:
    return f"{user_id}:{hash_value}"

def is_duplicate_progress(user_id: str, hash_value: str) -> bool:
    cache_key = get_cache_key(user_id, hash_value)
    if cache_key in _progress_cache:
        return time.time() - _progress_cache[cache_key] < _cache_expiry
    return False

def mark_progress_stored(user_id: str, hash_value: str):
    _progress_cache[get_cache_key(user_id, hash_value)] = time.time()

# ---------------- STORE PROGRESS ----------------
def store_progress(user_id: str, progress_data: Dict[str, Any], user_timezone=None) -> bool:
    """
    Store quiz progress results with duplicate prevention.
    """
    try:
        timestamp = time.time()
        
        # Create a stable hash for duplicate detection (remove timestamp)
        progress_copy = progress_data.copy()
        progress_copy.pop('timestamp', None)
        progress_hash = hashlib.md5(json.dumps(progress_copy, sort_keys=True).encode()).hexdigest()
        
        # Check for duplicate
        if is_duplicate_progress(user_id, progress_hash):
            print(f"[INFO] Duplicate progress detected for user {user_id}, skipping storage")
            return True
        
        mark_progress_stored(user_id, progress_hash)
        
        # Prepare local time info
        local_time_info = {}
        if user_timezone and TIMEZONE_AVAILABLE:
            local_time_info = {
                "local_time": format_timestamp_to_local(timestamp, user_timezone),
                "relative_time": get_relative_time(timestamp, user_timezone),
                "user_timezone": user_timezone
            }
        
        progress_json = json.dumps(progress_data)
        metadata = {
            "type": "progress",
            "progress_data": progress_json,
            "user_id": user_id,
            "timestamp": timestamp,
            "source": "quiz_result",
            "progress_hash": progress_hash,
            **local_time_info
        }
        
        emb = embed_text(progress_json)
        vector = {
            "id": f"{user_id}_progress_{int(timestamp)}_{progress_hash[:8]}",
            "values": emb,
            "metadata": metadata
        }
        
        index.upsert(vectors=[vector])
        print(f"[INFO] Stored progress for user {user_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] store_progress: {e}")
        return False

# ---------------- FETCH PROGRESS ----------------
def fetch_progress_from_pinecone(user_id: str, user_timezone=None):
    """
    Fetch and summarize user progress.
    
    Returns:
        dict: {
            "user_id": user_id,
            "progress": [...],  # List of progress entries
            "summary": {
                "total_attempts": int,
                "average_score": float,
                "average_accuracy": float,
                "topics_covered": list,
                "recent_activity": int
            }
        }
    """
    try:
        # Query for progress entries
        query_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = index.query(
            vector=query_vector,
            filter={"user_id": {"$eq": user_id}, "type": {"$eq": "progress"}}, 
            top_k=100, 
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
                print(f"[WARNING] Error parsing progress data: {e}")
                continue
        
        # Sort by timestamp (newest first)
        progress_list.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        if not progress_list:
            return {
                "user_id": user_id, 
                "progress": [], 
                "summary": {
                    "total_attempts": 0, 
                    "average_score": 0, 
                    "average_accuracy": 0,
                    "topics_covered": [],
                    "recent_activity": 0
                }
            }
        
        # Calculate summary statistics
        total_attempts = len(progress_list)
        total_score = sum(p.get("score", 0) for p in progress_list)
        total_accuracy = sum(p.get("accuracy", 0) for p in progress_list)
        
        average_score = round(total_score / total_attempts, 2)
        average_accuracy = round(total_accuracy / total_attempts, 2)
        
        # Get unique topics
        topics = set()
        for p in progress_list:
            topic = p.get("topic", "")
            if topic:
                topics.add(topic)
        
        # Count recent activity (last 7 days)
        one_week_ago = time.time() - (7 * 24 * 3600)
        recent_activity = len([p for p in progress_list if p.get('timestamp', 0) > one_week_ago])
        
        summary = {
            "total_attempts": total_attempts,
            "average_score": average_score,
            "average_accuracy": average_accuracy,
            "topics_covered": list(topics)[:10],
            "recent_activity": recent_activity
        }
        
        return {
            "user_id": user_id,
            "progress": progress_list,
            "summary": summary
        }
        
    except Exception as e:
        print(f"[ERROR] fetch_progress_from_pinecone: {e}")
        return {
            "user_id": user_id, 
            "progress": [], 
            "summary": {
                "total_attempts": 0, 
                "average_score": 0, 
                "average_accuracy": 0,
                "topics_covered": [],
                "recent_activity": 0
            }
        }

# ---------------- CLEANUP DUPLICATE PROGRESS ----------------
def cleanup_duplicate_progress(user_id: str):
    """
    Clean up duplicate progress entries for a user.
    Keeps only the most recent entry for each unique progress result.
    """
    try:
        query_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = index.query(
            vector=query_vector,
            filter={"user_id": {"$eq": user_id}, "type": {"$eq": "progress"}}, 
            top_k=200, 
            include_metadata=True
        )
        
        # Group by content hash
        hash_map = {}
        to_delete = []
        
        for match in getattr(results, "matches", []):
            meta = match.metadata or {}
            try:
                data = json.loads(meta.get("progress_data", "{}"))
                # Create a stable key without timestamp
                data_copy = data.copy()
                data_copy.pop('timestamp', None)
                data_copy.pop('local_time', None)
                data_copy.pop('relative_time', None)
                
                content_hash = hashlib.md5(json.dumps(data_copy, sort_keys=True).encode()).hexdigest()
                
                if content_hash not in hash_map:
                    hash_map[content_hash] = []
                hash_map[content_hash].append({
                    'id': match.id,
                    'timestamp': meta.get('timestamp', 0)
                })
            except Exception as e:
                print(f"[WARNING] Error parsing: {e}")
                continue
        
        # Keep only the most recent for each hash
        for hash_val, items in hash_map.items():
            if len(items) > 1:
                # Sort by timestamp (newest first)
                items.sort(key=lambda x: x['timestamp'], reverse=True)
                # Keep first (newest), delete rest
                to_delete.extend([item['id'] for item in items[1:]])
        
        if to_delete:
            print(f"[INFO] Deleting {len(to_delete)} duplicate progress entries for user {user_id}")
            for i in range(0, len(to_delete), 100):
                batch = to_delete[i:i+100]
                index.delete(ids=batch)
            return len(to_delete)
        
        print(f"[INFO] No duplicate progress entries found for user {user_id}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] cleanup_duplicate_progress: {e}")
        return 0

# ---------------- FORMAT PROGRESS FOR DISPLAY ----------------
def format_progress_for_display(progress_data: Dict[str, Any]) -> str:
    """Format progress data for nice UI display."""
    if not progress_data or not progress_data.get("progress"):
        return "📊 No progress data available yet."
    
    output = []
    summary = progress_data.get("summary", {})
    
    output.append("=" * 50)
    output.append("📊 PROGRESS SUMMARY")
    output.append("=" * 50)
    output.append(f"📝 Total Attempts: {summary.get('total_attempts', 0)}")
    output.append(f"⭐ Average Score: {summary.get('average_score', 0)}/5")
    output.append(f"🎯 Average Accuracy: {summary.get('average_accuracy', 0)}%")
    output.append(f"📚 Topics Covered: {len(summary.get('topics_covered', []))}")
    output.append(f"🔄 Recent Activity (7 days): {summary.get('recent_activity', 0)}")
    output.append("=" * 50)
    output.append("\n📋 RECENT ATTEMPTS")
    output.append("-" * 50)
    
    # Show recent 5 attempts
    recent = progress_data.get("progress", [])[:5]
    for i, attempt in enumerate(recent, 1):
        local_time = attempt.get('local_time', 'N/A')
        score = attempt.get('score', 0)
        total = attempt.get('total', 0)
        accuracy = attempt.get('accuracy', 0)
        topic = attempt.get('topic', 'Unknown')
        difficulty = attempt.get('difficulty', 'medium')
        
        output.append(f"\n{i}. {local_time}")
        output.append(f"   Topic: {topic}")
        output.append(f"   Difficulty: {difficulty.upper()}")
        output.append(f"   Score: {score}/{total} ({accuracy}%)")
    
    return "\n".join(output)