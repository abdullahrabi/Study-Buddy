# Sender.py - Pinecone Only Architecture
import os
import json
import time
import re
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import fitz  # PyMuPDF
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
# EMBEDDING FUNCTION
# ============================================

def embed_text(text: str):
    try:
        if not text or not text.strip():
            print("[WARNING] Empty text provided for embedding")
            hash_val = int(hashlib.md5(str(time.time()).encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        text = text.strip()
        if len(text) > 10000:
            text = text[:10000]
        
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        emb = None
        if result and result.embeddings and len(result.embeddings) > 0:
            if hasattr(result.embeddings[0], 'values'):
                emb = result.embeddings[0].values
            elif isinstance(result.embeddings[0], list):
                emb = result.embeddings[0]
            else:
                emb = list(result.embeddings[0])
        
        if emb is None:
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        if len(emb) != 768:
            if len(emb) < 768:
                emb = emb + [random.uniform(0.001, 0.002) for _ in range(768 - len(emb))]
            else:
                emb = emb[:768]
        
        norm = sum(v**2 for v in emb) ** 0.5
        if norm > 0:
            emb = [v / norm for v in emb]
        
        return emb
        
    except Exception as e:
        print(f"[ERROR] embed_text: {e}")
        if text:
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
        else:
            random.seed(int(time.time()))
        return [random.uniform(0.01, 0.02) for _ in range(768)]

# ============================================
# PINECONE CRUD OPERATIONS (All in One)
# ============================================

def generate_jwt(user_id: str, email: str) -> str:
    """Generate JWT token"""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt(token: str) -> dict:
    """Verify JWT token"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except:
        return None

def store_user(email: str, password: str) -> dict:
    """Store user in Pinecone"""
    # Check if user exists
    existing = find_user_by_email(email)
    if existing:
        return None
    
    # Hash password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create user_id
    user_id = f"user_{int(time.time())}"
    
    # Create embedding for auth record
    text = f"user_auth:{email}"
    embedding = embed_text(text)
    
    # Store in Pinecone
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
    
    return {
        "user_id": user_id,
        "email": email
    }

def find_user_by_email(email: str) -> dict:
    """Find user by email"""
    text = f"user_auth:{email}"
    embedding = embed_text(text)
    
    try:
        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            namespace="users",
            filter={"type": {"$eq": "user_auth"}}
        )
        
        for match in results.matches:
            if match.metadata and match.metadata.get('email') == email:
                return {
                    "user_id": match.metadata.get('user_id'),
                    "email": match.metadata.get('email'),
                    "password_hash": match.metadata.get('password_hash')
                }
        return None
    except Exception as e:
        print(f"Error finding user: {e}")
        return None

def verify_user(email: str, password: str) -> dict:
    """Verify user credentials"""
    user = find_user_by_email(email)
    if not user:
        return None
    
    if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return user
    return None

def store_conversation(user_id: str, question: str, answer: str, contexts: list):
    """Store conversation in Pinecone"""
    text = f"Q: {question}\nA: {answer}"
    embedding = embed_text(text)
    
    index.upsert(
        vectors=[{
            "id": f"{user_id}_{int(time.time())}",
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
    return True

def get_user_history(user_id: str, limit: int = 50) -> list:
    """Get user's chat history"""
    dummy_vector = [0.0] * 768
    
    results = index.query(
        vector=dummy_vector,
        top_k=limit,
        include_metadata=True,
        namespace="chat_history",
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": "chat_history"}
        }
    )
    
    history = []
    for match in results.matches:
        if match.metadata:
            history.append({
                "question": match.metadata.get('question', ''),
                "answer": match.metadata.get('answer', ''),
                "contexts": json.loads(match.metadata.get('contexts', '[]')),
                "timestamp": match.metadata.get('timestamp', '')
            })
    
    return history

def get_conversation_context(user_id: str, query: str, top_k: int = 3) -> list:
    """Get relevant context for user's query"""
    embedding = embed_text(query)
    
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="chat_history",
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": "chat_history"}
        }
    )
    
    contexts = []
    for match in results.matches:
        if match.metadata:
            contexts.append({
                "question": match.metadata.get('question', ''),
                "answer": match.metadata.get('answer', ''),
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
# RAG CONTEXT RETRIEVAL (Updated with user_id)
# ============================================

def retrieve_context(query: str, user_id: str = None, top_k: int = 5) -> tuple:
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
        
        matches = getattr(resp, "matches", [])
        print(f"  [DEBUG] Found {len(matches)} matches for: {query[:30]}...")
        
        if not matches:
            return "No relevant information found.", []
        
        contexts = []
        seen_texts = set()
        
        for match in matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            score = match.score or 0
            source = meta.get("type", meta.get("source", "unknown"))
            
            if text and score > 0.1 and text not in seen_texts:
                seen_texts.add(text)
                source_type = meta.get("type", meta.get("source", "unknown"))
                
                if len(text) > 800:
                    text = text[:800] + "..."
                
                contexts.append({
                    "text": text,
                    "source": source_type,
                    "score": score
                })
        
        contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_contexts = contexts[:3]
        
        if top_contexts:
            context_strings = [f"[From {c['source']}] {c['text']}" for c in top_contexts]
            result = "\n\n---\n\n".join(context_strings)
            return result, top_contexts
        
        return "No relevant information found.", []
        
    except Exception as e:
        print(f"[ERROR] retrieve_context: {e}")
        return "Sorry, couldn't retrieve information.", []

# ============================================
# TOOLS
# ============================================

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"
        page = wikipedia.page(search_results[0])
        return f"📚 Wikipedia: {page.title}\n\n{page.summary[:600]}...\n\n🔗 {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple pages found. Options: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool 
def web_search(query: str) -> str:
    """Performs a web search using Tavily."""
    try:
        if not TAVILY_API_KEY:
            return "Tavily API key not configured."
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, search_depth="advanced", max_results=5)
        
        if response and isinstance(response, dict):
            results = response.get('results', [])
            if results:
                formatted_results = []
                for i, result in enumerate(results[:3], 1):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')[:300]
                    url = result.get('url', '')
                    formatted_results.append(f"{i}. **{title}**\n   {content}...\n   🔗 {url}")
                return f"🌐 Web Search Results:\n\n" + "\n\n".join(formatted_results)
        
        return "No web search results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ============================================
# MAIN CHATBOT FUNCTION (Pinecone Only)
# ============================================

def get_gemini_response(user_input: str, history: list = None, user_id: str = None) -> Generator[str, None, None]:
    """StudyBuddy with Pinecone-only architecture."""
    try:
        system_prompt = """You are StudyBuddy 🤖, a smart research assistant and tutor.

Guidelines:
1. Use the available tools to look up information when needed
2. Always cite your sources when using tools
3. Be helpful, educational, and encouraging

Available tools:
- search_notes: Search the user's personal notes/documents
- search_wikipedia: Search Wikipedia for encyclopedia knowledge
- web_search: Search the web for current news and real-time information"""

        llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        @tool
        def search_notes(query: str) -> str:
            """Search the user's personal notes and documents."""
            try:
                context_str, contexts = retrieve_context(query, user_id, top_k=5)
                set_last_contexts(contexts)
                if context_str and "No relevant" not in context_str:
                    return f"📄 From your notes:\n\n{context_str}"
                return "No relevant information found in your notes."
            except Exception as e:
                return f"Error searching notes: {str(e)}"

        tools = [search_notes, search_wikipedia, web_search]

        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=MemorySaver(),
            system_prompt=system_prompt
        )

        messages = []
        if history:
            for h in history[-5:]:
                if h.get('role') == 'user':
                    messages.append(("user", h.get('message', '')))
                else:
                    messages.append(("assistant", h.get('message', '')))
        
        messages.append(("user", user_input))
        
        input_state = {"messages": messages}
        
        config = {
            "configurable": {
                "thread_id": user_id or "default_thread"
            }
        }
        
        collected_response = ""
        
        try:
            for chunk in agent.stream(
                input_state, 
                stream_mode="values",
                config=config
            ):
                last_msg = chunk["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"\n🔧 Using tools:")
                    for tool_call in last_msg.tool_calls:
                        print(f"   - {tool_call['name']}: {tool_call['args'].get('query', '')}")
                
                if hasattr(last_msg, 'content') and last_msg.content:
                    content = last_msg.content
                    collected_response += content
                    yield content
                    
        except Exception as stream_error:
            print(f"[WARNING] Agent streaming failed: {stream_error}")
            print("[INFO] Falling back to direct response...")
            
            context_str, contexts = retrieve_context(user_input, user_id, top_k=3)
            set_last_contexts(contexts)
            
            fallback_llm = ChatGroq(
                model="qwen/qwen3.6-27b",
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )
            
            fallback_prompt = f"""You are StudyBuddy, a helpful tutor. Answer based on context.

Context: {context_str if context_str else "No context available"}

User question: {user_input}"""
            
            response = fallback_llm.invoke(fallback_prompt)
            collected_response = response.content
            yield response.content
        
        # ============================================
        # SAVE TO PINECONE (No Vercel, No MongoDB)
        # ============================================
        if collected_response and user_input:
            contexts_used = get_last_contexts()
            context_texts = [c.get('text', '') for c in contexts_used if c.get('text')]
            
            try:
                # Store in Pinecone
                store_conversation(
                    user_id=user_id or "default_user",
                    question=user_input,
                    answer=collected_response,
                    contexts=context_texts
                )
                print(f"\n📊 Saved to Pinecone")
                
            except Exception as e:
                print(f"⚠️ Pinecone save error: {e}")

    except Exception as e:
        print(f"[ERROR] get_gemini_response: {e}")
        import traceback
        traceback.print_exc()
        yield f"I'm having trouble processing your request. Error: {str(e)}"

def get_studybuddy_response(user_input: str, history: list = None, user_id: str = None):
    """Alias for get_gemini_response"""
    return get_gemini_response(user_input, history, user_id)

# ============================================
# COMMAND LINE TESTING
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🤖 StudyBuddy with Pinecone Only")
    print("="*60)
    print("-"*60)
    
    # Login or Signup (using Pinecone only)
    while True:
        print("\n🔐 Authentication")
        print("1. Login")
        print("2. Signup")
        print("3. Exit")
        choice = input("Choose (1/2/3): ").strip()
        
        if choice == "3":
            exit()
        
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        
        if choice == "1":
            user = verify_user(email, password)
            if user:
                token = generate_jwt(user['user_id'], user['email'])
                print(f"✅ Logged in as: {user['email']}")
                user_id = user['user_id']
                break
            else:
                print("❌ Invalid credentials")
        elif choice == "2":
            user = store_user(email, password)
            if user:
                token = generate_jwt(user['user_id'], user['email'])
                print(f"✅ Signed up as: {user['email']}")
                user_id = user['user_id']
                break
            else:
                print("❌ User already exists")
        else:
            print("Invalid choice")
    
    print("\n💬 Type 'exit' to quit")
    print("-"*60)
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        print("\nStudyBuddy: ", end="", flush=True)
        full_response = ""
        
        for chunk in get_gemini_response(user_input, chat_history, user_id):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()
        chat_history.append({"role": "user", "message": user_input})
        chat_history.append({"role": "assistant", "message": full_response})