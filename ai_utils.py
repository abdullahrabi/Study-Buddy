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
def chunk_text(text: str, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """Improved chunking that respects sentence boundaries"""
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

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

# ---------------- STORE NOTES, QUIZZES, PROGRESS ----------------
def store_notes_and_progress(user_id: str, notes_text=None, quiz_data=None, result=None, user_timezone=None):
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
                
                emb = embed_text(chunk)
                vectors.append({
                    "id": f"{user_id}_notes_{int(timestamp)}_{i}",
                    "values": emb,
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
            
            emb = embed_text(quiz_json)
            vectors.append({
                "id": f"{user_id}_quiz_{int(timestamp)}",
                "values": emb,
                "metadata": metadata
            })

        if result:
            progress_json = json.dumps(result)
            metadata = {
                "type": "progress", 
                "progress_data": progress_json, 
                "user_id": user_id,
                "timestamp": timestamp,
                "source": "quiz_result",
                **local_time_info
            }
            
            emb = embed_text(progress_json)
            vectors.append({
                "id": f"{user_id}_progress_{int(timestamp)}",
                "values": emb,
                "metadata": metadata
            })

        if vectors:
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch)
            print(f"[INFO] Stored {len(vectors)} vectors for user {user_id}")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] store_notes_and_progress: {e}")
        return False

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
                temperature=0.7,
                max_output_tokens=2048,
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

# ---------------- CREATE FALLBACK QUIZ ----------------
def create_fallback_quiz(notes_text: str, difficulty: str, config: dict) -> Dict[str, Any]:
    """Create a fallback quiz when AI generation fails."""
    topic = notes_text[:100].split('.')[0].strip()
    if not topic or len(topic) < 10:
        topic = "Study Material"
    
    if difficulty == "difficult":
        return {
            "quiz": [
                {
                    "question": f"Which of the following statements about '{topic}' are correct? (Select ALL that apply)",
                    "options": {
                        "A": "The concept is widely accepted in scientific literature",
                        "B": "It has multiple interpretations depending on context", 
                        "C": "It was first proposed in the 21st century",
                        "D": "It applies only to specific cases"
                    },
                    "answer": "A,B",
                    "answer_type": "multiple"
                }
            ],
            "topic": topic,
            "difficulty": difficulty,
            "source": "fallback",
            "difficulty_config": config
        }
    else:
        return {
            "quiz": [
                {
                    "question": f"What is the main topic of '{topic}'?",
                    "options": {
                        "A": "General knowledge",
                        "B": "Science and technology", 
                        "C": "Arts and humanities",
                        "D": "Social sciences"
                    },
                    "answer": "A",
                    "answer_type": "single"
                }
            ],
            "topic": topic,
            "difficulty": difficulty,
            "source": "fallback",
            "difficulty_config": config
        }

# ---------------- ENHANCED QUIZ GENERATION ----------------
def generate_quiz_from_notes(notes_text: str, user_id: str = "default_user", 
                           num_questions: int = 5, difficulty: str = "medium",
                           user_timezone: str = None) -> Dict[str, Any]:
    """Generate quiz from notes OR topic with enhanced difficulty levels using latest SDK"""
    
    is_topic = len(notes_text.strip()) < 100 or " " not in notes_text.strip()
    
    difficulty_configs = {
        "easy": {
            "description": "Create basic recall questions that test simple facts and definitions.",
            "correct_options": "single",
            "tricky_level": "low",
            "distractors": "obviously wrong options"
        },
        "medium": {
            "description": "Create questions that require understanding and application of concepts.",
            "correct_options": "single",
            "tricky_level": "medium",
            "distractors": "plausible but incorrect options"
        },
        "hard": {
            "description": "Create challenging questions that require analysis and evaluation.",
            "correct_options": "mixed",
            "tricky_level": "high",
            "distractors": "partially correct options that seem right"
        },
        "difficult": {
            "description": "Create extremely challenging questions with multiple correct answers.",
            "correct_options": "multiple",
            "tricky_level": "very_high",
            "distractors": "options that are technically correct but not the best answer"
        }
    }
    
    config = difficulty_configs.get(difficulty, difficulty_configs["medium"])
    
    difficult_prompt_addon = """
FOR DIFFICULT LEVEL QUESTIONS:
1. Most questions (at least 60%) should have MULTIPLE correct answers
2. Include "Select ALL that apply" phrasing when multiple answers are correct
3. For single-answer questions, make them extremely tricky
""" if difficulty == "difficult" else ""
    
    notes_preview = notes_text[:3000].strip()
    if not notes_preview:
        notes_preview = "General knowledge and study material"
    
    prompt = f"""
You are an expert quiz creator. Create {num_questions} multiple-choice questions based on the following {"topic" if is_topic else "study notes"}:

{"TOPIC" if is_topic else "STUDY NOTES"}:
{notes_preview}

DIFFICULTY LEVEL: {difficulty.upper()}
{config['description']}
{difficult_prompt_addon}

IMPORTANT FORMATTING RULES:
1. Return ONLY valid JSON, no other text
2. For questions with MULTIPLE correct answers, format answer as comma-separated letters (e.g., "A,B")
3. For SINGLE correct answers, use a single letter (e.g., "A")
4. Always include exactly 4 options labeled A, B, C, D

Output valid JSON strictly in this format:
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
  "difficulty": "{difficulty}",
  "source": "{"topic" if is_topic else "notes"}"
}}
"""
    
    try:
        model = "gemini-2.5-flash"
        
        print(f"[DEBUG] Generating quiz with difficulty: {difficulty}")
        print(f"[DEBUG] Prompt length: {len(prompt)}")
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4048,
            )
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        text = text.strip()
        
        print(f"[DEBUG] Raw response length: {len(text)}")
        print(f"[DEBUG] Raw response preview: {text[:500]}...")
        
        # Clean the response
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        start_idx = text.find('{')
        if start_idx != -1:
            text = text[start_idx:]
        
        end_idx = text.rfind('}')
        if end_idx != -1:
            text = text[:end_idx + 1]
        
        print(f"[DEBUG] Cleaned response: {text[:500]}...")
        
        try:
            quiz_data = json.loads(text)
            
            if "quiz" not in quiz_data or not isinstance(quiz_data["quiz"], list):
                raise ValueError("Invalid quiz structure")
            
            # Post-process to ensure answer_type is set correctly
            for question in quiz_data.get("quiz", []):
                answer = question.get("answer", "").strip()
                if "," in answer:
                    question["answer_type"] = "multiple"
                    letters = [letter.strip().upper() for letter in answer.split(",") if letter.strip()]
                    question["answer"] = ",".join(sorted(set(letters))) if letters else ""
                else:
                    question["answer_type"] = "single"
                    question["answer"] = answer.upper() if answer else ""
            
            quiz_data["difficulty_config"] = config
            quiz_data["source"] = "topic" if is_topic else "notes"
            
            print(f"[INFO] Generated {len(quiz_data['quiz'])} {difficulty} questions")
            
            store_notes_and_progress(
                user_id=user_id,
                quiz_data=quiz_data,
                user_timezone=user_timezone
            )
            
            return quiz_data
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            return create_fallback_quiz(notes_text, difficulty, config)
        
    except Exception as e:
        print(f"[ERROR] generate_quiz_from_notes: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_quiz(notes_text, difficulty, config)

# ---------------- QUIZ EVALUATION ----------------
def evaluate_quiz_attempt(quiz_json, student_answers, user_id=None, user_timezone=None):
    """Evaluate quiz attempt with support for multiple correct answers"""
    correct = 0
    total = len(quiz_json.get("quiz", []))
    feedback = []
    
    for i, q in enumerate(quiz_json.get("quiz", [])):
        correct_answer_str = q.get("answer", "").strip().upper()
        answer_type = q.get("answer_type", "single")
        
        if answer_type == "multiple":
            correct_answers = set(sorted([ans.strip() for ans in correct_answer_str.split(",")]))
        else:
            correct_answers = set([correct_answer_str])
        
        user_answer_str = ""
        if i in student_answers:
            user_answer_str = student_answers[i]
        elif str(i) in student_answers:
            user_answer_str = student_answers[str(i)]
        
        if not isinstance(user_answer_str, str):
            user_answer_str = str(user_answer_str) if user_answer_str else ""
        
        user_answer_str = user_answer_str.strip().upper()
        
        if not user_answer_str:
            user_answers_set = set()
        elif "," in user_answer_str:
            user_answers_set = set(sorted([ans.strip() for ans in user_answer_str.split(",")]))
        else:
            user_answers_set = set([user_answer_str])
        
        is_correct = user_answers_set == correct_answers
        
        if is_correct:
            correct += 1
            if answer_type == "multiple":
                status = "✅ Correct (All answers)"
                explanation = f"Excellent! You correctly identified all {len(correct_answers)} correct answers"
            else:
                status = "✅ Correct"
                explanation = "Well done! You answered correctly."
        else:
            if answer_type == "multiple":
                missing = correct_answers - user_answers_set
                extra = user_answers_set - correct_answers
                
                if missing and extra:
                    status = "❌ Partially correct"
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. Missed: {', '.join(sorted(missing))}. Extra: {', '.join(sorted(extra))}"
                elif missing:
                    status = "❌ Partially correct"
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. You missed: {', '.join(sorted(missing))}"
                elif extra:
                    status = "❌ Partially correct"
                    explanation = f"Correct: {', '.join(sorted(correct_answers))}. You added extra: {', '.join(sorted(extra))}"
                else:
                    status = "❌ Incorrect"
                    explanation = f"Correct answers: {', '.join(sorted(correct_answers))}"
            else:
                status = "❌ Incorrect"
                explanation = f"The correct answer is {correct_answer_str}"
        
        feedback.append({
            "question": q["question"],
            "status": status,
            "correct_answers": list(sorted(correct_answers)),
            "user_answers": list(sorted(user_answers_set)),
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
        "difficulty_config": quiz_json.get("difficulty_config", {}),
        "partial_credit": round(partial_score, 2)
    }
    
    if user_id:
        store_notes_and_progress(user_id, result=progress_data, user_timezone=user_timezone)
    
    return progress_data

# ---------------- FORMAT QUIZ FOR DISPLAY ----------------
def format_quiz_for_display(quiz_data: Dict[str, Any]) -> str:
    """Format quiz data for nice display"""
    output = []
    output.append(f"📝 Quiz Topic: {quiz_data.get('topic', 'Unknown')}")
    output.append(f"📊 Difficulty: {quiz_data.get('difficulty', 'medium').upper()}")
    output.append("=" * 50)
    
    for i, question in enumerate(quiz_data.get("quiz", [])):
        output.append(f"\nQ{i+1}. {question['question']}")
        for option, text in question['options'].items():
            output.append(f"   {option}) {text}")
        
        answer_type = question.get('answer_type', 'single')
        if answer_type == 'multiple':
            output.append(f"   💡 This question has MULTIPLE correct answers (select ALL that apply)")
        else:
            output.append(f"   💡 This question has a SINGLE correct answer")
    
    return "\n".join(output)

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