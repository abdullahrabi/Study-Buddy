# rag_eval_trulens_final.py
import os
import time
import re
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ============================================
# CRITICAL: Set OTEL environment variables FIRST
# ============================================
os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"

# Now import the rest
from pinecone import Pinecone
from google import genai
from google.genai import types

from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.groq import Groq

from trulens.core import TruSession, Metric, FeedbackMode
from trulens.apps.app import TruApp
from trulens.dashboard import run_dashboard

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

print("=" * 70)
print("🚀 RAG EVALUATION WITH TRULENS")
print("=" * 70)
print(f"\n🐍 Python Version: {sys.version}")

# ============================================
# 1. CUSTOM EMBEDDING CLASS
# ============================================

class GeminiDirectEmbedding(BaseEmbedding):
    api_key: str
    model_name: str = "gemini-embedding-2"
    dimension: int = 768
    
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2", dimension: int = 768, **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, dimension=dimension, **kwargs)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def _get_query_embedding(self, query: str) -> list:
        return self._embed_text(query)
    
    def _get_text_embedding(self, text: str) -> list:
        return self._embed_text(text)
    
    async def _aget_query_embedding(self, query: str) -> list:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> list:
        return self._get_text_embedding(text)
    
    def _embed_text(self, text: str) -> list:
        try:
            if not text or not text.strip():
                return None
            if len(text) > 8000:
                text = text[:8000]
            
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=self.dimension)
            )
            
            if result and result.embeddings and len(result.embeddings) > 0:
                emb = result.embeddings[0].values
                norm = sum(v**2 for v in emb) ** 0.5
                if norm > 0:
                    return [v / norm for v in emb]
            return None
        except Exception as e:
            print(f"[ERROR] embed: {e}")
            return None
    
    @classmethod
    def class_name(cls) -> str:
        return "GeminiDirectEmbedding"

# ============================================
# 2. RAG SYSTEM
# ============================================

class DirectRAG:
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
        self.last_context = ""
        self.last_documents = []
        self.last_question = ""
        self.last_answer = ""
    
    def retrieve(self, query: str, top_k: int = 15) -> list:
        try:
            query_embedding = self.embed_model._embed_text(query)
            if not query_embedding:
                return []
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    documents.append({
                        'text': match.metadata['text'],
                        'score': match.score,
                        'metadata': match.metadata
                    })
            return documents
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []
    
    def get_context(self, question: str) -> str:
        try:
            documents = self.retrieve(question, top_k=5)
            if not documents:
                return "No documents retrieved"
            contexts = []
            for doc in documents[:5]:
                text = doc['text'][:300] if doc['text'] else "No text"
                contexts.append(f"[Score: {doc['score']:.3f}] {text}...")
            return "\n\n---\n\n".join(contexts)
        except Exception as e:
            return f"Error: {e}"
    
    def query(self, question: str) -> str:
        self.last_question = question
        try:
            documents = self.retrieve(question, top_k=10)
            self.last_documents = documents
            
            if not documents:
                return "No relevant documents found."
            
            top_docs = documents[:5]
            
            self.last_context = "\n\n---\n\n".join([
                f"[Score: {doc['score']:.3f}] {doc['text'][:500]}"
                for doc in top_docs
            ])
            
            context_text = "\n\n".join([doc['text'] for doc in top_docs])
            
            prompt = f"""You are a helpful assistant specializing in automata theory. Answer based on the context.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
            
            response = self.llm.complete(prompt)
            self.last_answer = str(response).strip()
            return self.last_answer
        except Exception as e:
            return f"Error: {e}"

# ============================================
# 3. METRIC EVALUATORS (Using Production Model)
# ============================================

def create_metric_evaluators():
    from groq import Groq as GroqClient
    client = GroqClient(api_key=GROQ_API_KEY)
    
    def clean_score(text: str) -> float:
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return 0.5
    
    # Use production model to avoid rate limits
    EVAL_MODEL = "llama-3.3-70b-versatile"
    
    def evaluate_relevance(input: str, output: str) -> float:
        prompt = f"""Rate the relevance of the answer to the question on a scale of 0-1.
        Return ONLY a number between 0 and 1.
        
        Question: {input}
        Answer: {output[:500] if output else "No answer"}
        
        Relevance score (0-1):"""
        
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return clean_score(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"  ⚠️ Relevance error: {e}")
            return 0.5
    
    def evaluate_quality(input: str, output: str) -> float:
        prompt = f"""Rate the quality of the answer on a scale of 0-1.
        Consider accuracy, completeness, and clarity.
        Return ONLY a number between 0 and 1.
        
        Question: {input}
        Answer: {output[:500] if output else "No answer"}
        
        Quality score (0-1):"""
        
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return clean_score(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"  ⚠️ Quality error: {e}")
            return 0.5
    
    return evaluate_relevance, evaluate_quality

# ============================================
# 4. MAIN EXECUTION
# ============================================

def main():
    print("\n🔧 Configuring...")
    
    embed_model = GeminiDirectEmbedding(
        api_key=GEMINI_API_KEY,
        model_name="gemini-embedding-2",
        dimension=768
    )
    
    llm = Groq(
        model="llama-3.3-70b-versatile",  # Production model
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"\n🔗 Connecting to Pinecone index '{INDEX_NAME}'...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise ValueError(f"❌ Index '{INDEX_NAME}' not found!")
    
    pinecone_index = pc.Index(INDEX_NAME)
    stats = pinecone_index.describe_index_stats()
    print(f"✅ Connected: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
    
    rag = DirectRAG(pinecone_index, embed_model, llm)
    print("✅ RAG system ready\n")
    
    eval_questions = [
        "What is an alphabet in automata theory?",
        "What is the symbol used to denote the empty string?",
        "What is the Kleene star operator used for?",
        "What is a formal language?",
        "What is the difference between ∅ and {ε}?",
        "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
        "List three software applications using automata.",
    ]
    
    print("📚 Loaded evaluation questions:")
    for i, q in enumerate(eval_questions, 1):
        print(f"  {i}. {q[:60]}...")
    
    print("\n" + "=" * 70)
    print("🚀 RUNNING EVALUATION")
    print("=" * 70)
    
    answers = []
    contexts = []
    latencies = []
    
    for i, q in enumerate(eval_questions, 1):
        print(f"\n📝 Q{i}: {q[:50]}...")
        print("-" * 40)
        
        context = rag.get_context(q)
        contexts.append([context])
        
        start_time = time.time()
        answer = rag.query(q)
        latency = time.time() - start_time
        answers.append(answer)
        latencies.append(latency)
        
        print(f"⏱️ Latency: {latency:.2f}s")
        print(f"🤖 Answer:\n{answer[:200]}...")
    
    print("\n" + "=" * 70)
    print("📊 CALCULATING METRICS")
    print("=" * 70)
    
    evaluate_relevance, evaluate_quality = create_metric_evaluators()
    
    print("🔄 Calculating metrics...")
    metrics_data = []
    for i, (q, a) in enumerate(zip(eval_questions, answers), 1):
        print(f"  Calculating metrics for Q{i}: {q[:40]}...")
        
        relevance_score = evaluate_relevance(q, a)
        quality_score = evaluate_quality(q, a)
        
        metrics_data.append({
            'question': q,
            'answer': a[:200] + "...",
            'relevance': relevance_score,
            'quality': quality_score,
            'latency': latencies[i-1]
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("trulens_results.csv", index=False)
    print("\n💾 Metrics saved to trulens_results.csv")
    
    print("\n" + "=" * 70)
    print("📊 METRICS SUMMARY")
    print("=" * 70)
    
    print("\n📈 Overall Averages:")
    print(f"  Relevance: {df_metrics['relevance'].mean():.3f}")
    print(f"  Quality: {df_metrics['quality'].mean():.3f}")
    print(f"  Average Latency: {df_metrics['latency'].mean():.2f}s")
    
    # ============================================
    # TRULENS EVALUATION - FIXED: Use feedbacks approach
    # ============================================
    
    print("\n" + "=" * 70)
    print("📊 SETTING UP TRULENS")
    print("=" * 70)
    
    # Delete old database
    if os.path.exists("default.sqlite"):
        try:
            os.remove("default.sqlite")
            print("🗑️ Deleted old database")
        except:
            pass
    
    # Create session
    session = TruSession()
    session.reset_database()
    print("✅ Session created")
    
    # ============================================
    # FIX: Create a wrapper class (like RAGApp in working version)
    # ============================================
    class RAGWrapper:
        def __init__(self, rag):
            self.rag = rag
        
        def respond(self, question: str):
            """Main method that TruLens will track - returns (answer, context)"""
            context = self.rag.get_context(question)
            answer = self.rag.query(question)
            return answer, context
    
    # Create wrapper
    rag_wrapper = RAGWrapper(rag)
    
    # Create metrics (using evaluate functions)
    f_relevance = (
        Metric(evaluate_relevance, name="Relevance")
        .on_input()
        .on_output()
    )
    
    f_quality = (
        Metric(evaluate_quality, name="Quality")
        .on_input()
        .on_output()
    )
    
    print("✅ Metrics defined")
    
    # ============================================
    # FIX: Use feedbacks parameter (NOT metrics)
    # ============================================
    tru_recorder = TruApp(
        rag_wrapper,
        app_name="RAG_Evaluation",
        app_version="v1.0",
        feedbacks=[f_relevance, f_quality],  # ← Use 'feedbacks', not 'metrics'
        main_method=rag_wrapper.respond      # ← Use wrapper method
    )
    
    print("✅ TruApp created with feedbacks")
    
    print("\n🔄 Running TruLens evaluation...")
    with tru_recorder:
        for i, q in enumerate(eval_questions, 1):
            print(f"  Evaluating {i}/{len(eval_questions)}: {q[:40]}...")
            rag_wrapper.respond(q)
    
    print("✅ TruLens evaluation complete")
    
    # ============================================
    # Wait for feedback to complete
    # ============================================
    
    print("\n⏳ Waiting for feedback results...")
    time.sleep(5)
    
    try:
        tru_recorder.wait_for_feedback_results()
        print("✅ Feedback results processed")
    except Exception as e:
        print(f"⚠️ wait error: {e}")
    
    time.sleep(3)
    
    try:
        session.force_flush()
        print("✅ Force flush completed")
    except Exception as e:
        print(f"⚠️ flush error: {e}")
    
    time.sleep(3)
    
    # ============================================
    # VERIFY RECORDS
    # ============================================
    
    print("\n🔄 Verifying records...")
    try:
        records_df, feedback_names = session.get_records_and_feedback(app_ids=["RAG_Evaluation"])
        if records_df is not None and len(records_df) > 0:
            print(f"✅ Found {len(records_df)} records in database")
            print(f"   Feedback columns: {feedback_names}")
            records_df.to_csv("trulens_records.csv", index=False)
            print("💾 Records saved to trulens_records.csv")
        else:
            print("⚠️ No records found in database")
            print("📁 Your metrics are saved in trulens_results.csv")
    except Exception as e:
        print(f"⚠️ Could not verify: {e}")
    
    # ============================================
    # LAUNCH DASHBOARD
    # ============================================
    
    print("\n" + "=" * 70)
    print("🚀 LAUNCHING TRULENS DASHBOARD")
    print("=" * 70)
    
    print("\n📊 Metrics visible in dashboard:")
    print("   ✓ Relevance")
    print("   ✓ Quality")
    
    print("\n💡 When dashboard opens:")
    print("   1. Click 'Leaderboard' tab first")
    print("   2. Then click 'Apps'")
    print("   3. Select 'RAG_Evaluation'")
    
    print("\n🌐 Opening dashboard at http://localhost:8501...")
    
    try:
        run_dashboard(session=session, port=8501)
    except Exception as e:
        print(f"⚠️ Dashboard error: {e}")
        print("\n💡 Manual launch:")
        print("   streamlit run trulens_eval/dashboard.py")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    print(f"""
📊 PERFORMANCE SUMMARY:
------------------------
- Total Questions: {len(df_metrics)}
- Avg Relevance: {df_metrics['relevance'].mean():.2f}
- Avg Quality: {df_metrics['quality'].mean():.2f}
- Avg Latency: {df_metrics['latency'].mean():.2f}s

📁 Files Generated:
- trulens_results.csv (All metrics)
- default.sqlite (TruLens database)

🌐 TruLens Dashboard: http://localhost:8501
    """)

if __name__ == "__main__":
    main()