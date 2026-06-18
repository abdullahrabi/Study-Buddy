import os
import json
import time
import re
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import fitz  # PyMuPDF
import docx2txt  # Word extraction
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from tavily import TavilyClient
import wikipedia
import requests

# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
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
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
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

# ---------------- FILE EXTRACTION HELPERS ----------------
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path).strip()

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# ---------------- CHUNKING & EMBEDDINGS ----------------
def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def embed_text(text: str):
    try:
        if not text or not text.strip():
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        text = text.strip()[:10000]
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        emb = None
        if result and result.embeddings and len(result.embeddings) > 0:
            emb = result.embeddings[0].values if hasattr(result.embeddings[0], 'values') else list(result.embeddings[0])
        
        if emb is None:
            return [random.uniform(0.01, 0.02) for _ in range(768)]
        
        if len(emb) != 768:
            emb = emb[:768] if len(emb) > 768 else emb + [random.uniform(0.001, 0.002) for _ in range(768 - len(emb))]
        
        if all(abs(v) < 0.0001 for v in emb):
            emb = [v + (random.random() * 0.0001 - 0.00005) for v in emb]
        
        return emb
    except Exception as e:
        print(f"[ERROR] embed_text: {e}")
        return [random.uniform(0.01, 0.02) for _ in range(768)]

# ---------------- DUPLICATE PREVENTION CACHE ----------------
_storage_cache = {}
_cache_expiry = 300

def get_cache_key(user_id: str, item_type: str, hash_value: str) -> str:
    return f"{user_id}:{item_type}:{hash_value}"

def is_duplicate(user_id: str, item_type: str, hash_value: str) -> bool:
    cache_key = get_cache_key(user_id, item_type, hash_value)
    if cache_key in _storage_cache:
        return time.time() - _storage_cache[cache_key] < _cache_expiry
    return False

def mark_stored(user_id: str, item_type: str, hash_value: str):
    _storage_cache[get_cache_key(user_id, item_type, hash_value)] = time.time()

# ---------------- STORE NOTES & QUIZZES ----------------
def store_notes_and_quizzes(user_id: str, notes_text=None, quiz_data=None, user_timezone=None):
    vectors = []
    timestamp = time.time()
    
    try:
        local_time_info = {}
        if user_timezone and TIMEZONE_AVAILABLE:
            local_time_info = {
                "local_time": format_timestamp_to_local(timestamp, user_timezone),
                "relative_time": get_relative_time(timestamp, user_timezone),
                "user_timezone": user_timezone
            }
        
        if quiz_data:
            quiz_copy = quiz_data.copy()
            quiz_copy.pop('timestamp', None)
            quiz_copy.pop('difficulty_config', None)
            content_hash = hashlib.md5(json.dumps(quiz_copy, sort_keys=True).encode()).hexdigest()
            
            if is_duplicate(user_id, "quiz", content_hash):
                print(f"[INFO] Duplicate quiz detected for user {user_id}, skipping storage")
                return True
            mark_stored(user_id, "quiz", content_hash)
        
        if notes_text:
            chunks = chunk_text(notes_text)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "type": "notes", 
                    "text": chunk, 
                    "user_id": user_id, 
                    "chunk_index": i,
                    "timestamp": timestamp,
                    "source": "uploaded_notes",
                    **local_time_info
                }
                vectors.append({
                    "id": f"{user_id}_notes_{int(timestamp)}_{i}",
                    "values": embed_text(chunk),
                    "metadata": metadata
                })
        
        if quiz_data:
            quiz_json = json.dumps(quiz_data)
            metadata = {
                "type": "quiz", 
                "quiz_data": quiz_json, 
                "user_id": user_id,
                "timestamp": timestamp,
                "source": "generated_quiz",
                **local_time_info
            }
            vectors.append({
                "id": f"{user_id}_quiz_{int(timestamp)}_{random.randint(1000, 9999)}",
                "values": embed_text(quiz_json),
                "metadata": metadata
            })
        
        if vectors:
            for i in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[i:i+100])
            print(f"[INFO] Stored {len(vectors)} vectors for user {user_id}")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] store_notes_and_quizzes: {e}")
        return False

# ==================== TOOLS (SAME AS CHATBOT) ====================

@tool
def search_wikipedia_tool(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        # Try using wikipedia library first
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"
            page = wikipedia.page(search_results[0])
            return f"📚 Wikipedia: {page.title}\n\n{page.summary[:800]}...\n\n🔗 {page.url}"
        except json.JSONDecodeError:
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
                        summary = page_data["extract"][:800]
                        return f"📚 Wikipedia: {title}\n\n{summary}...\n\n🔗 https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            return f"No Wikipedia articles found for '{query}'"
            
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple pages found for '{query}'. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No page found for '{query}'"
    except Exception as e:
        print(f"[WARNING] Wikipedia error: {e}")
        # Last fallback: Try simple API
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "title" in data and "extract" in data:
                    return f"📚 Wikipedia: {data['title']}\n\n{data['extract'][:800]}...\n\n🔗 {data.get('content_urls', {}).get('desktop', {}).get('page', '')}"
        except:
            pass
        return f"Error searching Wikipedia for '{query}'. Please try again."

@tool
def web_search_tool(query: str) -> str:
    """Performs a web search using Tavily and returns the top results."""
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
        
        return str(response) if response else "No web search results found"
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ==================== RESEARCH AGENT FOR QUIZ GENERATION ====================

def research_topic_for_quiz(topic: str) -> str:
    """
    Use the LangChain agent with tools to research a topic for quiz generation.
    This mirrors how the chatbot works.
    """
    try:
        print(f"[INFO] Researching topic with LangChain agent: {topic}")
        
        # System prompt for research agent
        system_prompt = """You are a research assistant that gathers comprehensive information about a topic for quiz creation.
        
        Your task:
        1. Use available tools to research the topic thoroughly
        2. Gather key facts, concepts, and important details
        3. Return a well-structured summary of the topic
        
        Available tools:
        - search_wikipedia: Search Wikipedia for encyclopedia knowledge
        - web_search: Search the web for current and comprehensive information
        
        Important: Provide a thorough summary that covers the main aspects of the topic.
        Include definitions, key concepts, important facts, and any relevant details.
        """

        # Initialize LLM
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        # Create tools list (using the same tools as chatbot)
        tools = [search_wikipedia_tool, web_search_tool]

        # Create agent
        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=MemorySaver(),
            system_prompt=system_prompt
        )

        # Prepare messages
        messages = [
            ("user", f"Please research this topic thoroughly and provide a comprehensive summary: {topic}")
        ]
        
        # Stream response with thread_id
        input_state = {"messages": messages}
        
        config = {
            "configurable": {
                "thread_id": f"quiz_research_{int(time.time())}"
            }
        }
        
        research_summary = ""
        
        for chunk in agent.stream(
            input_state, 
            stream_mode="values",
            config=config
        ):
            last_msg = chunk["messages"][-1]
            
            # Debug: Show tool usage
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"\n🔧 Research agent using tools:")
                for tool_call in last_msg.tool_calls:
                    print(f"   - {tool_call['name']}: {tool_call['args'].get('query', '')}")
            
            # Collect response
            if hasattr(last_msg, 'content') and last_msg.content:
                research_summary = last_msg.content
        
        print(f"[INFO] Research summary length: {len(research_summary)} characters")
        return research_summary
        
    except Exception as e:
        print(f"[ERROR] research_topic_for_quiz: {e}")
        import traceback
        traceback.print_exc()
        return f"Topic: {topic}. Create quiz based on general knowledge."

# ---------------- CREATE FALLBACK QUIZ ----------------
def create_fallback_quiz(topic: str, difficulty: str, config: dict) -> Dict[str, Any]:
    return {
        "quiz": [
            {
                "question": f"Which of the following statements about '{topic}' are correct? (Select ALL that apply)" if difficulty == "difficult" else f"What is the main topic of '{topic}'?",
                "options": {
                    "A": "The concept is widely accepted in scientific literature",
                    "B": "It has multiple interpretations depending on context",
                    "C": "It was first proposed in the 21st century",
                    "D": "It applies only to specific cases"
                } if difficulty == "difficult" else {
                    "A": "General knowledge",
                    "B": "Science and technology",
                    "C": "Arts and humanities",
                    "D": "Social sciences"
                },
                "answer": "A,B" if difficulty == "difficult" else "A",
                "answer_type": "multiple" if difficulty == "difficult" else "single"
            }
        ],
        "topic": topic[:100],
        "difficulty": difficulty,
        "source": "fallback",
        "difficulty_config": config
    }

# ---------------- GENERATE QUIZ ----------------
def generate_quiz_from_notes(notes_text: str, user_id: str = "default_user", 
                           num_questions: int = 5, difficulty: str = "medium",
                           user_timezone: str = None,
                           store_quiz: bool = True,
                           use_tools: bool = True) -> Dict[str, Any]:
    """
    Generate a quiz from notes or topic.
    For topics, uses the LangChain agent with tools for research (same as chatbot).
    """
    
    is_topic = len(notes_text.strip()) < 100 or " " not in notes_text.strip()
    
    # Use tools if it's a topic and tools are enabled
    if is_topic and use_tools:
        print(f"[INFO] Generating quiz from topic using LangChain agent research: {notes_text}")
        
        # Use the research agent to get comprehensive information
        research_context = research_topic_for_quiz(notes_text.strip())
        
        # Use the research context as notes preview
        notes_preview = research_context[:3000].strip()
        source_label = "topic with AI research"
        tools_used = True
    else:
        notes_preview = notes_text[:3000].strip()
        source_label = "study notes"
        tools_used = False
    
    if not notes_preview:
        notes_preview = "General knowledge and study material"
    
    difficulty_configs = {
        "easy": {
            "description": "Basic recall questions testing simple facts.",
            "correct_options": "single"
        },
        "medium": {
            "description": "Understanding and application questions.",
            "correct_options": "single"
        },
        "hard": {
            "description": "Analysis and evaluation questions.",
            "correct_options": "mixed"
        },
        "difficult": {
            "description": "Extremely challenging questions with multiple correct answers.",
            "correct_options": "multiple"
        }
    }
    
    config = difficulty_configs.get(difficulty, difficulty_configs["medium"])
    
    difficult_prompt_addon = """
FOR DIFFICULT LEVEL QUESTIONS:
1. Most questions (at least 60%) should have MULTIPLE correct answers
2. For single-answer questions, make them extremely tricky
""" if difficulty == "difficult" else ""
    
    prompt = f"""
You are an expert quiz creator. Create {num_questions} multiple-choice questions based on the following {source_label}:

CONTEXT:
{notes_preview}

DIFFICULTY LEVEL: {difficulty.upper()}
{config['description']}
{difficult_prompt_addon}

IMPORTANT FORMATTING RULES:
1. Return ONLY valid JSON, no other text
2. For MULTIPLE correct answers, format answer as "A,B"
3. For SINGLE correct answers, use a single letter "A"
4. Always include exactly 4 options labeled A, B, C, D

Output valid JSON in this format:
{{
  "quiz": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      }},
      "answer": "A,B",
      "answer_type": "multiple"
    }}
  ],
  "topic": "{notes_preview[:100]}",
  "difficulty": "{difficulty}"
}}
"""
    
    try:
        print(f"[INFO] Generating quiz with difficulty: {difficulty}")
        
        if not GROQ_API_KEY:
            print("[ERROR] GROQ_API_KEY not found!")
            return create_fallback_quiz(notes_text[:100], difficulty, config)
        
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.7,
            groq_api_key=GROQ_API_KEY
        )
        
        response = llm.invoke(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        text = re.sub(r'```json\s*|```\s*', '', text.strip())
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        
        quiz_data = json.loads(text)
        
        if "quiz" not in quiz_data or not isinstance(quiz_data["quiz"], list):
            raise ValueError("Invalid quiz structure")
        
        for q in quiz_data["quiz"]:
            answer = q.get("answer", "").strip()
            if "," in answer:
                q["answer_type"] = "multiple"
                letters = [a.strip().upper() for a in answer.split(",") if a.strip()]
                q["answer"] = ",".join(sorted(set(letters)))
            else:
                q["answer_type"] = "single"
                q["answer"] = answer.upper() if answer else ""
        
        quiz_data["difficulty_config"] = config
        quiz_data["source"] = "topic_with_agent" if tools_used else ("topic" if is_topic else "notes")
        quiz_data["tools_used"] = tools_used
        
        print(f"[INFO] Generated {len(quiz_data['quiz'])} questions")
        
        if store_quiz:
            store_notes_and_quizzes(
                user_id=user_id,
                quiz_data=quiz_data,
                user_timezone=user_timezone
            )
        
        return quiz_data
        
    except Exception as e:
        print(f"[ERROR] generate_quiz_from_notes: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_quiz(notes_text[:100], difficulty, config)

# ---------------- CONVENIENCE FUNCTION ----------------
def generate_quiz_from_topic(topic: str, user_id: str = "default_user",
                           num_questions: int = 5, difficulty: str = "medium",
                           user_timezone: str = None, store_quiz: bool = True) -> Dict[str, Any]:
    return generate_quiz_from_notes(
        notes_text=topic,
        user_id=user_id,
        num_questions=num_questions,
        difficulty=difficulty,
        user_timezone=user_timezone,
        store_quiz=store_quiz,
        use_tools=True
    )

# ---------------- FORMAT QUIZ ----------------
def format_quiz_for_display(quiz_data: Dict[str, Any]) -> str:
    output = [
        f"📝 Topic: {quiz_data.get('topic', 'Unknown')}",
        f"📊 Difficulty: {quiz_data.get('difficulty', 'medium').upper()}",
        f"🔍 Tools Used: {'Yes ✅' if quiz_data.get('tools_used', False) else 'No'}",
        f"📄 Source: {quiz_data.get('source', 'unknown')}",
        "=" * 50
    ]
    
    for i, q in enumerate(quiz_data.get("quiz", [])):
        output.append(f"\nQ{i+1}. {q['question']}")
        for opt, text in q['options'].items():
            output.append(f"   {opt}) {text}")
        
        answer_type = q.get('answer_type', 'single')
        if answer_type == 'multiple':
            output.append(f"   💡 This question has MULTIPLE correct answers (select ALL that apply)")
        else:
            output.append(f"   💡 This question has a SINGLE correct answer")
    
    return "\n".join(output)

# ---------------- QUIZ EVALUATION (MODIFIED WITH PROGRESS STORAGE) ----------------
def evaluate_quiz_attempt(quiz_json, student_answers, user_id=None, user_timezone=None, store_progress=True):
    """
    Evaluate a quiz attempt and optionally store progress.
    
    Args:
        quiz_json: The quiz data containing questions and answers
        student_answers: Dictionary of student's answers (question index -> answer)
        user_id: User identifier for storing progress
        user_timezone: User's timezone for timestamp formatting
        store_progress: Whether to store progress in Pinecone
    
    Returns:
        progress_data: Dictionary containing evaluation results
    """
    correct = 0
    total = len(quiz_json.get("quiz", []))
    feedback = []
    
    for i, q in enumerate(quiz_json.get("quiz", [])):
        correct_answers = set(q.get("answer", "").upper().split(",")) if q.get("answer") else set()
        user_answer = student_answers.get(i, student_answers.get(str(i), "")).strip().upper()
        user_answers = set(user_answer.split(",")) if user_answer else set()
        
        is_correct = user_answers == correct_answers
        if is_correct:
            correct += 1
        
        answer_type = q.get("answer_type", "single")
        
        if is_correct:
            status = "✅ Correct"
            explanation = "Well done! You answered correctly."
        else:
            if answer_type == "multiple":
                missing = correct_answers - user_answers
                extra = user_answers - correct_answers
                if missing and extra:
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. Missed: {', '.join(sorted(missing))}. Extra: {', '.join(sorted(extra))}"
                elif missing:
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. You missed: {', '.join(sorted(missing))}"
                elif extra:
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. You added extra: {', '.join(sorted(extra))}"
                else:
                    explanation = f"Correct answers: {', '.join(sorted(correct_answers))}"
            else:
                explanation = f"The correct answer is {', '.join(sorted(correct_answers))}"
            
            status = "❌ Incorrect" if answer_type == "single" else "❌ Partially correct"
        
        feedback.append({
            "question": q["question"],
            "status": status,
            "correct_answers": list(correct_answers),
            "user_answers": list(user_answers),
            "answer_type": answer_type,
            "explanation": explanation,
            "is_correct": is_correct
        })
    
    accuracy = round((correct / total * 100), 2) if total else 0
    
    partial_score = 0
    for fb in feedback:
        if fb["answer_type"] == "multiple" and not fb["is_correct"]:
            user_set = set(fb["user_answers"])
            correct_set = set(fb["correct_answers"])
            correct_selected = len(user_set.intersection(correct_set))
            incorrect_selected = len(user_set - correct_set)
            if correct_selected > 0:
                partial = (correct_selected - (incorrect_selected * 0.5)) / len(correct_set)
                if partial > 0:
                    partial_score += partial
    
    progress_data = {
        "score": correct,
        "adjusted_score": round(correct + partial_score, 2),
        "total": total,
        "accuracy": accuracy,
        "feedback": feedback,
        "timestamp": time.time(),
        "difficulty": quiz_json.get("difficulty", "medium"),
        "topic": quiz_json.get("topic", "Unknown"),
        "source": quiz_json.get("source", "unknown"),
        "tools_used": quiz_json.get("tools_used", False),
        "difficulty_config": quiz_json.get("difficulty_config", {}),
        "partial_credit": round(partial_score, 2)
    }
    
    # Store progress if requested and user_id is provided
    if store_progress and user_id:
        try:
            print(f"[INFO] Storing progress for user: {user_id}")
            
            # Import Progress module here to avoid circular imports
            from Progress import store_progress as store_progress_data
            
            success = store_progress_data(
                user_id=user_id,
                progress_data=progress_data,
                user_timezone=user_timezone
            )
            
            if success:
                print(f"[INFO] Progress stored successfully for user {user_id}")
            else:
                print(f"[WARNING] Failed to store progress for user {user_id}")
                
        except ImportError as e:
            print(f"[WARNING] Progress module not found: {e}")
            # Try direct storage as fallback
            try:
                store_progress_direct(user_id, progress_data, user_timezone)
            except Exception as e2:
                print(f"[ERROR] Direct storage failed: {e2}")
                
        except Exception as e:
            print(f"[ERROR] Failed to store progress: {e}")
            import traceback
            traceback.print_exc()
    
    return progress_data


# ---------------- DIRECT PROGRESS STORAGE (FALLBACK) ----------------
def store_progress_direct(user_id: str, progress_data: Dict[str, Any], user_timezone=None) -> bool:
    """
    Direct storage of progress data to Pinecone (fallback if Progress module is not available).
    """
    try:
        timestamp = time.time()
        
        # Create a stable hash for duplicate detection
        progress_copy = progress_data.copy()
        progress_copy.pop('timestamp', None)
        progress_hash = hashlib.md5(json.dumps(progress_copy, sort_keys=True).encode()).hexdigest()
        
        # Check for duplicate in cache
        if is_duplicate(user_id, "progress", progress_hash):
            print(f"[INFO] Duplicate progress detected for user {user_id}, skipping storage")
            return True
        
        mark_stored(user_id, "progress", progress_hash)
        
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
        
        # Generate embedding
        emb = embed_text(progress_json)
        
        # Ensure embedding has correct dimensions
        if len(emb) != 768:
            print(f"[WARNING] Embedding length {len(emb)} != 768, adjusting")
            if len(emb) < 768:
                emb = emb + [random.uniform(0.001, 0.002) for _ in range(768 - len(emb))]
            else:
                emb = emb[:768]
        
        # Create vector
        vector = {
            "id": f"{user_id}_progress_{int(timestamp)}_{progress_hash[:8]}",
            "values": emb,
            "metadata": metadata
        }
        
        # Upsert to Pinecone
        index.upsert(vectors=[vector])
        print(f"[INFO] Direct storage: Progress vector stored for user {user_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Direct storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------- STORE PROGRESS (WRAPPER FOR BACKWARD COMPATIBILITY) ----------------
def store_progress(user_id: str, progress_data: Dict[str, Any], user_timezone=None) -> bool:
    """
    Wrapper function to store progress. This function is called from the Progress module.
    """
    return store_progress_direct(user_id, progress_data, user_timezone)