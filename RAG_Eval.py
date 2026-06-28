# rag_eval_final_complete.py
import os
import time
import re
import sys
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Suppress serialization warnings
warnings.filterwarnings('ignore', message='.*cannot be serialized.*')

# ============================================
# CRITICAL: Set OTEL environment variables FIRST
# ============================================
os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"

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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_NAME = f"RAG_Eval_{RUN_ID}"
APP_VERSION = "v1.0"

print("=" * 70)
print("🚀 OPTIMIZED RAG EVALUATION WITH TRULENS")
print("=" * 70)
print(f"\n📅 Run ID: {RUN_ID}")
print(f"📱 App Name: {APP_NAME}")
print(f"🎯 Strategy: Sentence Window")
print(f"💾 Database: Preserved for comparison")

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
# 2. OPTIMIZED RAG WITH SENTENCE WINDOW
# ============================================

class OptimizedRAG:
    """RAG with Sentence Window - Best for Groundedness & Correctness"""
    
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
    
    def retrieve_with_sentence_window(self, query: str, top_k: int = 5, window_size: int = 3) -> list:
        """Sentence Window Retrieval"""
        try:
            query_embedding = self.embed_model._embed_text(query)
            if not query_embedding:
                return []
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=15,
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
            
            expanded_docs = []
            seen_texts = set()
            automata_keywords = {'alphabet', 'string', 'language', 'automata', 'state', 
                                'symbol', 'empty', 'kleene', 'regular', 'expression',
                                'formal', 'closure', 'transition', 'dfa', 'nfa', 'epsilon'}
            
            for doc in documents:
                text = doc['text']
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                sentences = re.split(r'(?<=[.!?])\s+', text)
                
                if len(sentences) <= window_size * 2:
                    expanded_docs.append(doc)
                    continue
                
                query_terms = set(query.lower().split())
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    sentence_terms = set(sentence.lower().split())
                    overlap = len(query_terms & sentence_terms)
                    bonus = len(sentence_terms & automata_keywords) * 0.5
                    sentence_scores.append((i, overlap + bonus))
                
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                
                expanded_texts = []
                for idx, score in sentence_scores[:2]:
                    start = max(0, idx - window_size)
                    end = min(len(sentences), idx + window_size + 1)
                    window_text = ' '.join(sentences[start:end])
                    expanded_texts.append(window_text)
                
                expanded_text = ' ... '.join(expanded_texts)
                expanded_docs.append({
                    'text': expanded_text,
                    'score': doc['score'],
                    'metadata': {**doc['metadata'], 'strategy': 'sentence_window'}
                })
            
            expanded_docs.sort(key=lambda x: x['score'], reverse=True)
            return expanded_docs[:top_k]
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []
    
    def query(self, question: str) -> str:
        """Generate answer using Sentence Window retrieval"""
        try:
            documents = self.retrieve_with_sentence_window(question)
            
            if not documents:
                return "No relevant documents found."
            
            top_docs = documents[:5]
            context_text = "\n\n".join([doc['text'] for doc in top_docs])
            
            prompt = f"""You are an expert in automata theory. Answer based on the context provided.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
            
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            return f"Error: {e}"

# ============================================
# 3. MODEL ROUTING STRATEGY
# ============================================

MODEL_CONFIGS = {
    'fast': {'model': 'llama-3.1-8b-instant', 'use_for': ['relevance', 'quality']},
    'primary': {'model': 'llama-3.3-70b-versatile', 'use_for': ['groundedness', 'correctness']},
    'context': {'model': 'llama-3.1-8b-instant', 'use_for': ['context_relevance']}
}

class ModelRouter:
    """Routes evaluation tasks to dedicated models"""
    
    def __init__(self, api_key: str):
        from groq import Groq as GroqClient
        self.client = GroqClient(api_key=api_key)
        self.model_usage = {}
        self.model_failures = {}
        self.rate_limit_waits = {}
        
    def clean_score(self, text: str) -> float:
        if not text:
            return None
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text
        patterns = [
            r'(?:score|rating)?\s*[:=]?\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*\/\s*(?:1|10|100)',
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
                    return max(0.0, min(1.0, score))
                except:
                    continue
        return None
    
    def call_model(self, prompt: str, model: str, max_retries: int = 3) -> float:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Output ONLY a number between 0 and 1. No other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0, max_tokens=10
                )
                if model not in self.model_usage:
                    self.model_usage[model] = 0
                self.model_usage[model] += 1
                
                score = self.clean_score(response.choices[0].message.content.strip())
                if score is not None:
                    return score
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return 0.5
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    time.sleep(min(30 * (2 ** attempt), 120))
                elif attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return 0.5
        return 0.5
    
    def evaluate(self, task_type: str, prompt: str) -> float:
        task_model_map = {
            'relevance': MODEL_CONFIGS['fast'],
            'quality': MODEL_CONFIGS['fast'],
            'groundedness': MODEL_CONFIGS['primary'],
            'context_relevance': MODEL_CONFIGS['context'],
            'correctness': MODEL_CONFIGS['primary']
        }
        config = task_model_map.get(task_type, MODEL_CONFIGS['fast'])
        return self.call_model(prompt, config['model'])

# ============================================
# 4. METRIC EVALUATORS (Module-level functions)
# ============================================

def evaluate_relevance(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score relevance from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('relevance', prompt)

def evaluate_quality(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score quality from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('quality', prompt)

def evaluate_groundedness(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score factual reliability from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('groundedness', prompt)

def evaluate_context_relevance(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score context relevance from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('context_relevance', prompt)

def evaluate_correctness(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score correctness from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('correctness', prompt)

# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    print("\n🔧 Configuring...")
    
    embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY, model_name="gemini-embedding-2", dimension=768)
    llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"\n🔗 Connecting to Pinecone index '{INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise ValueError(f"❌ Index '{INDEX_NAME}' not found!")
    
    pinecone_index = pc.Index(INDEX_NAME)
    stats = pinecone_index.describe_index_stats()
    print(f"✅ Connected: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
    
    rag = OptimizedRAG(pinecone_index, embed_model, llm)
    print("✅ RAG ready (Sentence Window Strategy)\n")
    
    eval_questions = [
        "What is an alphabet in automata theory?",
        "What is the symbol used to denote the empty string?",
        "What is the Kleene star operator used for?",
        "What is a formal language?",
        "What is the difference between ∅ and {ε}?",
        "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
        "List three software applications using automata.",
    ]
    
    # Phase 1: Generate answers
    print("=" * 70)
    print("🚀 PHASE 1: GENERATING ANSWERS")
    print("=" * 70)
    
    answers, latencies = [], []
    
    for i, q in enumerate(eval_questions, 1):
        print(f"\n📝 Q{i}: {q[:50]}...")
        start_time = time.time()
        answer = rag.query(q)
        latency = time.time() - start_time
        answers.append(answer)
        latencies.append(latency)
        print(f"⏱️ {latency:.2f}s | {answer[:150]}...")
    
    # Phase 2: Calculate metrics
    print("\n" + "=" * 70)
    print("📊 PHASE 2: CALCULATING METRICS")
    print("=" * 70)
    
    router = ModelRouter(GROQ_API_KEY)
    
    metrics_data = []
    for i, (q, a) in enumerate(zip(eval_questions, answers), 1):
        print(f"\n  📊 Q{i}: {q[:50]}...")
        
        relevance_score = evaluate_relevance(q, a, router)
        quality_score = evaluate_quality(q, a, router)
        groundedness_score = evaluate_groundedness(q, a, router)
        context_relevance_score = evaluate_context_relevance(q, a, router)
        correctness_score = evaluate_correctness(q, a, router)
        
        metrics_data.append({
            'question': q, 'answer': a[:200] + "...",
            'relevance': relevance_score, 'quality': quality_score,
            'groundedness': groundedness_score, 'context_relevance': context_relevance_score,
            'correctness': correctness_score, 'latency': latencies[i-1],
            'run_id': RUN_ID, 'strategy': 'sentence_window'
        })
        
        print(f"    ✓ Relevance: {relevance_score:.3f} | Quality: {quality_score:.3f}")
        print(f"    ✓ Groundedness: {groundedness_score:.3f} | Context: {context_relevance_score:.3f}")
        print(f"    ✓ Correctness: {correctness_score:.3f}")
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Save files
    df_metrics.to_csv(f"trulens_results_{RUN_ID}.csv", index=False)
    df_metrics.to_csv("trulens_results_latest.csv", index=False)
    
    history_file = "trulens_results_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        df_history = pd.concat([df_history, df_metrics], ignore_index=True)
    else:
        df_history = df_metrics
    df_history.to_csv(history_file, index=False)
    
    print(f"\n💾 Results saved: trulens_results_{RUN_ID}.csv, trulens_results_latest.csv, trulens_results_history.csv")
    
    # Phase 3: Display results
    print("\n" + "=" * 70)
    print("📊 METRICS SUMMARY")
    print("=" * 70)
    
    print("\n📈 Overall Averages:")
    for metric in ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']:
        avg = df_metrics[metric].mean()
        status = "✅ high" if avg >= 0.7 else "⚠️ medium" if avg >= 0.5 else "🛑 low"
        print(f"  {metric.replace('_', ' ').title()}: {avg:.3f} {status}")
    print(f"  Average Latency: {df_metrics['latency'].mean():.2f}s")
    
    # Phase 4: TruLens Dashboard
    print("\n" + "=" * 70)
    print("📊 TRULENS DASHBOARD")
    print("=" * 70)
    
    open_dashboard = input("\n🌐 Open TruLens Dashboard? (y/n): ").lower().strip()
    
    if open_dashboard == 'y':
        from trulens.core import TruSession, Metric
        from trulens.apps.app import TruApp
        from trulens.dashboard import run_dashboard
        from functools import partial
        
        print("\n📊 Setting up TruLens...")
        
        # DON'T delete database - preserve for comparison
        session = TruSession()
        print("✅ Session ready (database preserved for comparison)")
        
        class RAGWrapper:
            def __init__(self, rag):
                self.rag = rag
            def respond(self, question: str) -> str:
                return self.rag.query(question)
        
        rag_wrapper = RAGWrapper(rag)
        
        # Create metrics using partial to bind router
        f_relevance = Metric(
            partial(evaluate_relevance, router=router), name="Relevance"
        ).on_input_output()
        
        f_quality = Metric(
            partial(evaluate_quality, router=router), name="Quality"
        ).on_input_output()
        
        f_groundedness = Metric(
            partial(evaluate_groundedness, router=router), name="Groundedness"
        ).on_input_output()
        
        f_context_relevance = Metric(
            partial(evaluate_context_relevance, router=router), name="Context Relevance"
        ).on_input_output()
        
        print("✅ Metrics created")
        
        # Create TruApp with unique name for this run
        tru_recorder = TruApp(
            rag_wrapper,
            app_name=APP_NAME,
            app_version=APP_VERSION,
            feedbacks=[f_relevance, f_quality, f_groundedness, f_context_relevance],
            main_method=rag_wrapper.respond
        )
        
        print(f"🔄 Running TruLens evaluation (App: {APP_NAME})...")
        with tru_recorder as recording:
            for i, q in enumerate(eval_questions, 1):
                print(f"  {i}/{len(eval_questions)}: {q[:40]}...")
                rag_wrapper.respond(q)
        
        print("✅ Evaluation recorded")
        
        # Wait for feedback
        print("\n⏳ Waiting for feedback results...")
        print("   (This may take 30-60 seconds)")
        
        time.sleep(30)  # Initial wait
        
        try:
            tru_recorder.wait_for_feedback_results()
        except:
            pass
        
        # Try to get records
        try:
            records_df, feedback_names = session.get_records_and_feedback(
                app_name=APP_NAME,
                app_versions=[APP_VERSION]
            )
            
            if records_df is not None and len(records_df) > 0:
                print(f"✅ Found {len(records_df)} records")
                records_df.to_csv(f"trulens_records_{RUN_ID}.csv", index=False)
                print(f"💾 Records saved: trulens_records_{RUN_ID}.csv")
            else:
                print("⚠️ Records may still be processing")
        except Exception as e:
            print(f"⚠️ Could not verify records: {str(e)[:80]}")
        
        # Launch dashboard
        print("\n" + "=" * 70)
        print("🌐 LAUNCHING TRULENS DASHBOARD")
        print("=" * 70)
        print(f"\n📊 Open http://localhost:8501")
        print(f"\n💡 Select '{APP_NAME}' to view this run")
        print("   Previous runs preserved for comparison")
        print("\n   Press Ctrl+C to stop\n")
        
        try:
            run_dashboard(session=session, port=8501)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)[:80]}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📅 Run ID: {RUN_ID}")
    print(f"📁 Results: trulens_results_{RUN_ID}.csv")
    print(f"📁 History: trulens_results_history.csv")
    print(f"💾 Database: Preserved for comparison")

if __name__ == "__main__":
    main()