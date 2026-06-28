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
from llama_index.llms.groq import Groq as LlamaGroq

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

print("=" * 70)
print("🚀 RAG EVALUATION WITH TRULENS DASHBOARD")
print("=" * 70)
print(f"\n🐍 Python Version: {sys.version}")

# ============================================
# 1. CUSTOM EMBEDDING CLASS
# ============================================

class GeminiDirectEmbedding(BaseEmbedding):
    """Direct Gemini embedding without LlamaIndex wrapper"""
    
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
        """Get embeddings from Gemini API with normalization"""
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
    """RAG system using Gemini embeddings, Pinecone vector DB, and Groq LLM"""
    
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
        self.last_context = ""
        self.last_documents = []
        self.last_question = ""
        self.last_answer = ""
    
    def retrieve(self, query: str, top_k: int = 15) -> list:
        """Retrieve relevant documents from Pinecone"""
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
    
    def query(self, question: str) -> str:
        """Generate answer using RAG"""
        self.last_question = question
        try:
            documents = self.retrieve(question, top_k=10)
            self.last_documents = documents
            
            if not documents:
                return "No relevant documents found."
            
            top_docs = documents[:5]
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
# 3. MODEL ROUTING STRATEGY
# ============================================

MODEL_CONFIGS = {
    'fast': {
        'model': 'llama-3.1-8b-instant',
        'speed': 560,
        'use_for': ['relevance', 'quality']
    },
    'primary': {
        'model': 'llama-3.3-70b-versatile',
        'speed': 280,
        'use_for': ['groundedness', 'correctness']
    },
    'context': {
        'model': 'llama-3.1-8b-instant',
        'speed': 560,
        'use_for': ['context_relevance']
    }
}

class ModelRouter:
    """Routes evaluation tasks with fixed parsing for all models"""
    
    def __init__(self, api_key: str):
        from groq import Groq as GroqClient
        self.client = GroqClient(api_key=api_key)
        self.model_usage = {}
        self.model_failures = {}
        self.rate_limit_waits = {}
        
    def clean_score(self, text: str) -> float:
        """Extract numerical score from model response"""
        if not text:
            return None
        
        # Remove <think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        
        if not cleaned:
            cleaned = text
        
        # Try to find any number in the cleaned text
        patterns = [
            r'(?:score|rating|relevance|quality|groundedness|correctness|context)?\s*[:=]?\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*\/\s*(?:1|10|100)',
            r'(?:^|\s)(\d+\.?\d*)(?:\s|$)',
            r'(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned.lower())
            if matches:
                try:
                    score = float(matches[0])
                    if score > 1 and score <= 10:
                        score = score / 10
                    elif score > 10 and score <= 100:
                        score = score / 100
                    elif score > 100:
                        score = 0.5
                    return max(0.0, min(1.0, score))
                except:
                    continue
        
        # Keyword fallback
        if any(word in cleaned.lower() for word in ['excellent', 'perfect', 'outstanding']):
            return 0.95
        elif any(word in cleaned.lower() for word in ['very good', 'high']):
            return 0.85
        elif any(word in cleaned.lower() for word in ['good', 'adequate']):
            return 0.75
        elif any(word in cleaned.lower() for word in ['fair', 'moderate', 'average']):
            return 0.5
        elif any(word in cleaned.lower() for word in ['poor', 'low']):
            return 0.25
        elif any(word in cleaned.lower() for word in ['very poor', 'terrible', 'incorrect']):
            return 0.1
        
        print(f"    ⚠️ Could not parse score from: '{cleaned[:150]}...'")
        return None
    
    def call_model_with_retry(self, prompt: str, model: str, task_type: str, max_retries: int = 3) -> float:
        """Call a specific model with intelligent retry logic"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a scoring system. Output ONLY a number between 0 and 1. No explanations, no thinking, no analysis. Just the number."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                
                if model not in self.model_usage:
                    self.model_usage[model] = 0
                self.model_usage[model] += 1
                
                response_text = response.choices[0].message.content.strip()
                score = self.clean_score(response_text)
                
                if score is not None:
                    return score
                else:
                    if attempt < max_retries - 1:
                        print(f"    ⚠️ Unparseable response, retrying... (attempt {attempt + 1})")
                        prompt = f"OUTPUT ONLY A NUMBER BETWEEN 0 AND 1. NO OTHER TEXT.\n\n{prompt}"
                        continue
                    else:
                        print(f"    ⚠️ Could not parse score after {max_retries} attempts")
                        return 0.5
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate_limit" in error_msg:
                    wait_time = min(30 * (2 ** attempt), 120)
                    print(f"    ⚠️ Rate limit on {model}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if model not in self.rate_limit_waits:
                        self.rate_limit_waits[model] = 0
                    self.rate_limit_waits[model] += wait_time
                    
                elif "server_error" in error_msg or "overloaded" in error_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f"    ⚠️ Server issue. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                else:
                    print(f"    ❌ Error: {str(e)[:100]}")
                    if model not in self.model_failures:
                        self.model_failures[model] = 0
                    self.model_failures[model] += 1
                    
                    if attempt == max_retries - 1:
                        return 0.5
                    time.sleep(3)
        
        return 0.5
    
    def evaluate(self, task_type: str, prompt: str) -> float:
        """Route task to its dedicated model"""
        
        task_model_map = {
            'relevance': MODEL_CONFIGS['fast'],
            'quality': MODEL_CONFIGS['fast'],
            'groundedness': MODEL_CONFIGS['primary'],
            'context_relevance': MODEL_CONFIGS['context'],
            'correctness': MODEL_CONFIGS['primary']
        }
        
        config = task_model_map.get(task_type, MODEL_CONFIGS['fast'])
        model = config['model']
        
        print(f"      🎯 {task_type.upper()}: {model}")
        
        return self.call_model_with_retry(prompt, model, task_type)

# ============================================
# 4. METRIC EVALUATORS
# ============================================

def create_metric_evaluators(router: ModelRouter):
    """Create evaluation functions with improved prompts"""
    
    def evaluate_relevance(input: str, output: str) -> float:
        prompt = f"""Score the relevance of this answer from 0 to 1.
Output ONLY the number.

Question: {input[:300]}
Answer: {output[:300] if output else "None"}

Score:"""
        return router.evaluate('relevance', prompt)
    
    def evaluate_quality(input: str, output: str) -> float:
        prompt = f"""Score the quality of this answer from 0 to 1.
Consider accuracy, completeness, and clarity.
Output ONLY the number.

Question: {input[:300]}
Answer: {output[:300] if output else "None"}

Score:"""
        return router.evaluate('quality', prompt)
    
    def evaluate_groundedness(input: str, output: str) -> float:
        prompt = f"""Score the factual reliability of this answer from 0 to 1.
Check for accuracy and absence of false information.
Output ONLY the number.

Question: {input[:300]}
Answer: {output[:300] if output else "None"}

Score:"""
        return router.evaluate('groundedness', prompt)
    
    def evaluate_context_relevance(input: str, output: str) -> float:
        prompt = f"""Score how contextually relevant this answer is from 0 to 1.
Output ONLY the number.

Question: {input[:300]}
Answer: {output[:300] if output else "None"}

Score:"""
        return router.evaluate('context_relevance', prompt)
    
    def evaluate_correctness(input: str, output: str) -> float:
        prompt = f"""Score the correctness of this answer from 0 to 1.
Output ONLY the number.

Question: {input[:300]}
Answer: {output[:300] if output else "None"}

Score:"""
        return router.evaluate('correctness', prompt)
    
    return {
        'relevance': evaluate_relevance,
        'quality': evaluate_quality,
        'groundedness': evaluate_groundedness,
        'context_relevance': evaluate_context_relevance,
        'correctness': evaluate_correctness
    }

# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    print("\n🔧 Configuring...")
    
    embed_model = GeminiDirectEmbedding(
        api_key=GEMINI_API_KEY,
        model_name="gemini-embedding-2",
        dimension=768
    )
    
    llm = LlamaGroq(
        model="llama-3.1-8b-instant",
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
    
    # ============================================
    # PHASE 1: GENERATE ANSWERS
    # ============================================
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 1: GENERATING ANSWERS")
    print("=" * 70)
    
    answers = []
    latencies = []
    
    for i, q in enumerate(eval_questions, 1):
        print(f"\n📝 Q{i}: {q[:50]}...")
        print("-" * 40)
        
        start_time = time.time()
        answer = rag.query(q)
        latency = time.time() - start_time
        answers.append(answer)
        latencies.append(latency)
        
        print(f"⏱️ Latency: {latency:.2f}s")
        print(f"🤖 Answer:\n{answer[:200]}...")
    
    # ============================================
    # PHASE 2: CALCULATE METRICS
    # ============================================
    
    print("\n" + "=" * 70)
    print("📊 PHASE 2: CALCULATING METRICS")
    print("=" * 70)
    
    router = ModelRouter(GROQ_API_KEY)
    evaluators = create_metric_evaluators(router)
    
    print("\n🎯 MODEL STRATEGY:")
    print("  • Relevance, Quality, Context → Llama-3.1-8B (560 T/s)")
    print("  • Groundedness, Correctness → Llama-3.3-70B (Best quality)")
    
    print("\n🔄 Calculating metrics...")
    metrics_data = []
    
    for i, (q, a) in enumerate(zip(eval_questions, answers), 1):
        print(f"\n  📊 Q{i}: {q[:50]}...")
        
        relevance_score = evaluators['relevance'](q, a)
        quality_score = evaluators['quality'](q, a)
        groundedness_score = evaluators['groundedness'](q, a)
        context_relevance_score = evaluators['context_relevance'](q, a)
        correctness_score = evaluators['correctness'](q, a)
        
        metrics_data.append({
            'question': q,
            'answer': a[:200] + "...",
            'relevance': relevance_score,
            'quality': quality_score,
            'groundedness': groundedness_score,
            'context_relevance': context_relevance_score,
            'correctness': correctness_score,
            'latency': latencies[i-1]
        })
        
        print(f"    ✓ Relevance: {relevance_score:.3f} | Quality: {quality_score:.3f}")
        print(f"    ✓ Groundedness: {groundedness_score:.3f} | Context: {context_relevance_score:.3f}")
        print(f"    ✓ Correctness: {correctness_score:.3f}")
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("trulens_results.csv", index=False)
    print("\n💾 Metrics saved to trulens_results.csv")
    
    # ============================================
    # PHASE 3: DISPLAY RESULTS
    # ============================================
    
    print("\n" + "=" * 70)
    print("📊 METRICS SUMMARY")
    print("=" * 70)
    
    print("\n📈 Overall Averages:")
    for metric in ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']:
        avg = df_metrics[metric].mean()
        status = "✅ high" if avg >= 0.7 else "⚠️ medium" if avg >= 0.5 else "🛑 low"
        print(f"  {metric.replace('_', ' ').title()}: {avg:.2f} {status}")
    print(f"  Average Latency: {df_metrics['latency'].mean():.2f}s")
    
    print("\n📋 Per-Question Breakdown:")
    print("-" * 70)
    for i, row in df_metrics.iterrows():
        print(f"\nQ{i+1}: {row['question'][:60]}...")
        for metric in ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']:
            score = row[metric]
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "🛑"
            print(f"  {status} {metric.replace('_', ' ').title()}: {score:.3f}")
        print(f"  ⏱️ Latency: {row['latency']:.2f}s")
    
    print("\n" + "=" * 70)
    print("📊 MODEL USAGE")
    print("=" * 70)
    for model, count in router.model_usage.items():
        failures = router.model_failures.get(model, 0)
        success_rate = ((count - failures) / count * 100) if count > 0 else 0
        print(f"  • {model}: {count} calls, {failures} failures ({success_rate:.0f}% success)")
    
    if router.rate_limit_waits:
        print("\n⚠️ Rate Limit Waits:")
        for model, wait_time in router.rate_limit_waits.items():
            print(f"  • {model}: {wait_time:.0f}s total wait time")
    else:
        print("✅ No rate limits encountered!")
    
    # ============================================
    # PHASE 4: TRULENS DASHBOARD (OPTIONAL)
    # ============================================
    
    print("\n" + "=" * 70)
    print("📊 TRULENS DASHBOARD")
    print("=" * 70)
    
    print("\n💡 Your metrics are saved to trulens_results.csv")
    print("   The dashboard provides interactive visualizations")
    open_dashboard = input("\n🌐 Open TruLens Dashboard? (y/n): ").lower().strip()
    
    if open_dashboard == 'y':
        from trulens.core import TruSession, Metric
        from trulens.apps.app import TruApp
        from trulens.dashboard import run_dashboard
        
        print("\n📊 Setting up TruLens evaluation...")
        
        # Clean start
        if os.path.exists("default.sqlite"):
            try:
                os.remove("default.sqlite")
                print("🗑️ Deleted old database")
            except:
                pass
        
        session = TruSession()
        session.reset_database()
        print("✅ Session created")
        
        # Create wrapper
        class RAGWrapper:
            def __init__(self, rag):
                self.rag = rag
            def respond(self, question: str) -> str:
                return self.rag.query(question)
        
        rag_wrapper = RAGWrapper(rag)
        
        # Create feedback metrics
        f_relevance = Metric(evaluators['relevance'], name="Relevance").on_input_output()
        f_quality = Metric(evaluators['quality'], name="Quality").on_input_output()
        f_groundedness = Metric(evaluators['groundedness'], name="Groundedness").on_input_output()
        f_context_relevance = Metric(evaluators['context_relevance'], name="Context Relevance").on_input_output()
        
        print("✅ Metrics created")
        
        # Create TruApp
        tru_recorder = TruApp(
            rag_wrapper,
            app_name="RAG_Evaluation",
            app_version="v1.0",
            feedbacks=[f_relevance, f_quality, f_groundedness, f_context_relevance],
            main_method=rag_wrapper.respond
        )
        
        print("🔄 Running TruLens evaluation...")
        with tru_recorder as recording:
            for i, q in enumerate(eval_questions, 1):
                print(f"  {i}/{len(eval_questions)}: {q[:40]}...")
                rag_wrapper.respond(q)
        
        print("✅ Evaluation recorded")
        
        # CRITICAL: Wait for feedback results with verification
        print("\n⏳ Waiting for feedback results to process...")
        print("   (This typically takes 30-90 seconds)")
        
        feedback_ready = False
        max_wait = 180  # 3 minutes max
        wait_interval = 15  # Check every 15 seconds
        waited = 0
        
        while waited < max_wait and not feedback_ready:
            time.sleep(wait_interval)
            waited += wait_interval
            
            try:
                # Try to wait for feedback
                tru_recorder.wait_for_feedback_results()
                
                # Check database
                records_df, feedback_names = session.get_records_and_feedback(app_ids=["RAG_Evaluation"])
                
                if records_df is not None and len(records_df) > 0:
                    # Check if feedback columns have actual values
                    feedback_cols = [col for col in records_df.columns if any(name in col for name in feedback_names)]
                    has_values = False
                    
                    for col in feedback_cols:
                        if col in records_df.columns and records_df[col].notna().any():
                            has_values = True
                            sample_val = records_df[col].dropna().iloc[0]
                            print(f"   ✓ {col}: {sample_val:.3f}")
                    
                    if has_values:
                        feedback_ready = True
                        print(f"\n✅ Feedback results ready! ({waited}s)")
                        print(f"   Records: {len(records_df)}")
                        
                        # Save records
                        records_df.to_csv("trulens_records.csv", index=False)
                        print("   💾 Records saved to trulens_records.csv")
                    else:
                        print(f"   ⏳ Records exist but feedback still calculating... ({waited}s)")
                else:
                    print(f"   ⏳ Waiting for records... ({waited}s)")
                    
            except Exception as e:
                print(f"   ⏳ Still processing... ({waited}s)")
        
        if not feedback_ready:
            print("\n⚠️ Feedback processing is taking longer than expected")
            print("   Dashboard may show records without feedback initially")
            print("   Feedback may appear after a few minutes in the dashboard")
        
        # Force flush
        try:
            session.force_flush()
            time.sleep(5)
        except:
            pass
        
        # Final verification
        print("\n🔍 Final database check...")
        try:
            records_df, feedback_names = session.get_records_and_feedback(app_ids=["RAG_Evaluation"])
            if records_df is not None and len(records_df) > 0:
                print(f"✅ Database ready: {len(records_df)} records")
                print(f"   Columns: {list(records_df.columns)[:5]}...")
            else:
                print("⚠️ No records found, but dashboard may still work")
        except:
            pass
        
        # Launch dashboard
        print("\n" + "=" * 70)
        print("🌐 LAUNCHING TRULENS DASHBOARD")
        print("=" * 70)
        print("\n📊 Open http://localhost:8501 in your browser")
        print("\n💡 Tips if metrics don't show immediately:")
        print("   1. Click 'Leaderboard' tab first")
        print("   2. Select 'RAG_Evaluation' from app dropdown")
        print("   3. Wait 1-2 minutes and refresh the page")
        print("   4. Feedback calculates in the background")
        print("\n   Press Ctrl+C to stop the dashboard\n")
        
        try:
            run_dashboard(session=session, port=8501)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"\n⚠️ Dashboard error: {e}")
            print("\n💡 Alternative launch methods:")
            print("   1. Run: python -m trulens.dashboard")
            print("   2. Or check: http://localhost:8501")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    metrics_avg = {
        'Relevance': df_metrics['relevance'].mean(),
        'Quality': df_metrics['quality'].mean(),
        'Groundedness': df_metrics['groundedness'].mean(),
        'Context Relevance': df_metrics['context_relevance'].mean(),
        'Correctness': df_metrics['correctness'].mean()
    }
    
    best_metric = max(metrics_avg, key=metrics_avg.get)
    worst_metric = min(metrics_avg, key=metrics_avg.get)
    
    print(f"""
📊 PERFORMANCE SUMMARY:
------------------------
✅ Best: {best_metric} ({metrics_avg[best_metric]:.2f})
⚠️ Needs Work: {worst_metric} ({metrics_avg[worst_metric]:.2f})

📁 Files Generated:
- trulens_results.csv (Complete metrics)
{f"- trulens_records.csv (TruLens records)" if open_dashboard == 'y' else ""}
{f"- default.sqlite (TruLens database)" if open_dashboard == 'y' else ""}
    """)

if __name__ == "__main__":
    main()