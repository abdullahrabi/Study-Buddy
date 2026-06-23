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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

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

# ---------------- TIMEZONE UTILITIES ----------------
def get_user_timezone(timezone_str=None):
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
    if not TIMEZONE_AVAILABLE:
        return ["UTC"]
    return [
        'UTC', 'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
        'America/Toronto', 'America/Vancouver', 'Europe/London', 'Europe/Paris',
        'Europe/Berlin', 'Europe/Moscow', 'Asia/Kolkata', 'Asia/Singapore',
        'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Dubai', 'Australia/Sydney',
        'Australia/Perth', 'Pacific/Auckland', 'Africa/Johannesburg',
        'America/Sao_Paulo', 'America/Mexico_City'
    ]

def get_current_time_in_timezone(timezone_str=None):
    current_time = time.time()
    return format_timestamp_to_local(current_time, timezone_str)

def format_timestamp_for_display(timestamp, user_timezone=None):
    if not timestamp:
        return "N/A"
    local_time = format_timestamp_to_local(timestamp, user_timezone)
    relative_time = get_relative_time(timestamp, user_timezone)
    return f"{local_time} ({relative_time})"

# ---------------- IMPROVED EMBEDDING FUNCTION ----------------
def embed_text(text: str):
    try:
        if not text or not text.strip():
            print("[WARNING] Empty text provided for embedding")
            # Return a random vector with some pattern
            import hashlib
            hash_val = int(hashlib.md5(str(time.time()).encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        text = text.strip()
        if len(text) > 10000:
            text = text[:10000]
        
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[text],
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
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
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        if len(emb) != 768:
            if len(emb) < 768:
                emb = emb + [random.uniform(0.001, 0.002) for _ in range(768 - len(emb))]
            else:
                emb = emb[:768]
        
        # Normalize the vector
        norm = sum(v**2 for v in emb) ** 0.5
        if norm > 0:
            emb = [v / norm for v in emb]
        
        return emb
        
    except Exception as e:
        print(f"[ERROR] embed_text: {e}")
        import hashlib
        if text:
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
        else:
            random.seed(int(time.time()))
        return [random.uniform(0.01, 0.02) for _ in range(768)]

# ---------------- IMPROVED RAG CONTEXT RETRIEVAL ----------------
def retrieve_context(query: str, user_id: str = None, top_k: int = 5) -> str:
    """
    Retrieve relevant context from Pinecone with improved handling.
    Returns context string or helpful message if no context found.
    """
    try:
        # Generate embedding for the query
        q_emb = embed_text(query)
        
        # Prepare filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = {"$eq": user_id}
        
        # Query Pinecone
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
        
        # Debug: Check what's returned
        matches = getattr(resp, "matches", [])
        print(f"  [DEBUG] Found {len(matches)} matches for: {query[:30]}...")
        
        if not matches:
            print(f"  [DEBUG] No matches found at all!")
            return "No relevant information found in the knowledge base."
        
        # Collect contexts with lower threshold
        contexts = []
        seen_texts = set()
        source_info = []
        
        for match in matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            score = match.score or 0
            source = meta.get("type", meta.get("source", "unknown"))
            
            # DEBUG: Show scores
            print(f"  [DEBUG] Score: {score:.3f} - Source: {source} - Text length: {len(text)}")
            
            # Lower threshold to 0.1 to catch more results
            if text and score > 0.1 and text not in seen_texts:
                seen_texts.add(text)
                source_type = meta.get("type", meta.get("source", "unknown"))
                
                # Truncate if too long
                if len(text) > 800:
                    text = text[:800] + "..."
                
                context_text = f"[From {source_type}] {text}"
                contexts.append((score, context_text))
                source_info.append(source_type)
        
        # Sort by score (highest first)
        contexts.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 3 contexts
        top_contexts = [ctx[1] for ctx in contexts[:3]]
        
        if top_contexts:
            result = "\n\n---\n\n".join(top_contexts)
            print(f"  [DEBUG] Returning {len(top_contexts)} contexts, total length: {len(result)}")
            return result
        
        # If no context found with score > 0.1, try without score filter
        print(f"  [DEBUG] No high-score contexts found, trying fallback...")
        
        # Fallback: Take any context with text
        fallback_contexts = []
        for match in matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                if len(text) > 800:
                    text = text[:800] + "..."
                fallback_contexts.append(f"[From {meta.get('type', 'unknown')}] {text}")
                if len(fallback_contexts) >= 3:
                    break
        
        if fallback_contexts:
            return "\n\n---\n\n".join(fallback_contexts)
        
        return "No relevant information found in the knowledge base. I'll try to answer based on general knowledge."
        
    except Exception as e:
        print(f"[ERROR] retrieve_context: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I couldn't retrieve information from the knowledge base. Please try again."


# ---------------- CHAT HISTORY FUNCTIONS ----------------
def save_chat_history(user_id: str, chat_history: list, user_timezone=None):
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
        return True
    except Exception as e:
        print(f"[ERROR] save_chat_history: {e}")
        return False

def fetch_chat_history(user_id: str, user_timezone=None):
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

# ---------------- IMPROVED WIKIPEDIA TOOL ----------------
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic. Use this for encyclopedia knowledge."""
    try:
        # Try using wikipedia library first
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"
            page = wikipedia.page(search_results[0])
            return f"📚 Wikipedia: {page.title}\n\n{page.summary[:600]}...\n\n🔗 {page.url}"
        except:
            # Fallback: Use direct API request
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get("query", {}).get("search"):
                title = data["query"]["search"][0]["title"]
                # Get the summary
                params = {
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "format": "json"
                }
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if "extract" in page_data:
                        summary = page_data["extract"][:600]
                        return f"📚 Wikipedia: {title}\n\n{summary}...\n\n🔗 https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            return f"No Wikipedia articles found for '{query}'"
            
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple pages found for '{query}'. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No page found for '{query}'"
    except Exception as e:
        print(f"[WARNING] Wikipedia error: {e}")
        return f"Error searching Wikipedia for '{query}'. Please try again."

# ---------------- IMPROVED TAVILY TOOL ----------------
@tool 
def web_search(query: str) -> str:
    """Performs a web search using Tavily and returns the top results. Use this for current news and real-time information."""
    try:
        if not TAVILY_API_KEY:
            return "Tavily API key not configured. Please set TAVILY_API_KEY in .env file."
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        
        # Format the response nicely
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
        
        return "No web search results found for your query."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ---------------- MAIN CHATBOT FUNCTION (FIXED) ----------------
def get_gemini_response(user_input: str, history: list = None, user_id: str = None) -> Generator[str, None, None]:
    """StudyBuddy with RAG + Wikipedia + Tavily using LangChain Agent."""
    try:
        # System prompt
        system_prompt = """You are StudyBuddy 🤖, a smart research assistant and tutor.

Guidelines:
1. Use the available tools to look up information when needed
2. You can make multiple tool calls (either together or in sequence)
3. Only look up information when you are sure of what you want
4. Always cite your sources when using tools
5. Be helpful, educational, and encouraging
6. When you get tool results, summarize them clearly for the user

Available tools:
- search_notes: Search the user's personal notes/documents
- search_wikipedia: Search Wikipedia for encyclopedia knowledge
- web_search: Search the web for current news and real-time information

Remember: You are a tutor helping students learn, so explain concepts clearly and ask follow-up questions when appropriate."""

        # Use Llama 4 Scout (available and has good tool support)
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        # Create RAG tool with user_id
        @tool
        def search_notes(query: str) -> str:
            """Search the user's personal notes and documents."""
            try:
                context = retrieve_context(query, user_id, top_k=5)
                if context:
                    return f"📄 From your notes:\n\n{context}"
                return "No relevant information found in your notes."
            except Exception as e:
                return f"Error searching notes: {str(e)}"

        tools = [search_notes, search_wikipedia, web_search]

        # Create agent with tools
        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=MemorySaver(),
            system_prompt=system_prompt
        )

        # Prepare messages
        messages = []
        
        # Add chat history
        if history:
            for h in history[-5:]:
                if h.get('role') == 'user':
                    messages.append(("user", h.get('message', '')))
                else:
                    messages.append(("assistant", h.get('message', '')))
        
        # Add current query
        messages.append(("user", user_input))
        
        # Stream response with thread_id
        input_state = {"messages": messages}
        
        config = {
            "configurable": {
                "thread_id": user_id or "default_thread"
            }
        }
        
        # Try streaming with agent
        try:
            for chunk in agent.stream(
                input_state, 
                stream_mode="values",
                config=config
            ):
                last_msg = chunk["messages"][-1]
                
                # Debug: Show tool usage
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"\n🔧 Using tools:")
                    for tool_call in last_msg.tool_calls:
                        print(f"   - {tool_call['name']}: {tool_call['args'].get('query', '')}")
                
                # Yield response
                if hasattr(last_msg, 'content') and last_msg.content:
                    yield last_msg.content
                    
        except Exception as stream_error:
            print(f"[WARNING] Agent streaming failed: {stream_error}")
            print("[INFO] Falling back to direct response...")
            
            # Fallback: Direct response without tools
            # Get context from notes
            context = retrieve_context(user_input, user_id, top_k=3)
            
            # Use a simpler model for fallback
            fallback_llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )
            
            fallback_prompt = f"""You are StudyBuddy, a helpful tutor. Answer the user's question based on the context.

Context from notes:
{context if context else "No context available"}

User question: {user_input}

Provide a clear, educational response. If the question is about current events, suggest using the web search feature for the most current information."""
            
            response = fallback_llm.invoke(fallback_prompt)
            yield response.content

    except Exception as e:
        print(f"[ERROR] get_gemini_response: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback: Simple response
        yield f"I'm having trouble processing your request. Please try again or rephrase your question. Error: {str(e)}"

# ---------------- COMPATIBILITY WRAPPER ----------------
def get_studybuddy_response(user_input: str, history: list = None, user_id: str = None):
    """Alias for get_gemini_response"""
    return get_gemini_response(user_input, history, user_id)