# Complete file - Fixed for your TruLens version
from Chatbot import (
    retrieve_context,
    save_chat_history,
    fetch_chat_history,
    get_gemini_response,
)
import os
import numpy as np
import pandas as pd
import json
import time
import re
import random
import warnings
from datetime import datetime
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import fitz  # PyMuPDF
import docx2txt  # Word extraction
import serpapi
import langgraph
import langchain
import wikipedia
from groq import Groq
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
import gc
import atexit

# ============================================
# 1. PATCH TORCH (for Python 3.13)
# ============================================
if not hasattr(torch, 'Tensor'):
    class Tensor:
        pass
    torch.Tensor = Tensor
    print("✅ Patched torch.Tensor")

# ============================================
# 2. IMPORT TRULENS - CORRECT WAY WITH TruApp
# ============================================
try:
    from trulens.core import TruSession, Metric
    from trulens.apps.app import TruApp
    TRULENS_AVAILABLE = True
    print("✅ TruLens imported successfully with TruApp!")
except ImportError as e:
    print(f"⚠️ TruLens import error: {e}")
    TRULENS_AVAILABLE = False
    # Fallback classes
    class TruSession:
        def __init__(self): pass
        def run_dashboard(self): print("Dashboard not available")
        def get_records(self): return []
        def reset_database(self): print("Database reset (simulated)")
        def record(self, **kwargs): pass
    class Metric:
        def __init__(self, func): self.func = func
        def on_input(self): return self
        def on_output(self): return self
        def on_context(self, **kwargs): return self
        def aggregate(self, func): return self
    class TruApp:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ============================================
# 3. LOAD ENV VARIABLES
# ============================================
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

# ============================================
# 4. RESET TRULENS DATABASE (CRITICAL FIX!)
# ============================================
print("\n🧹 Resetting TruLens database to ensure clean run...")

if TRULENS_AVAILABLE:
    try:
        session = TruSession()
        # Try different reset methods
        try:
            session.reset_database()
        except:
            try:
                session.reset()
            except:
                print("⚠️ Could not reset database, continuing...")
        print("✅ TruLens ready!")
    except Exception as e:
        print(f"⚠️ Could not reset database: {e}")
        session = TruSession()
else:
    session = TruSession()
    print("⚠️ TruLens not available - running without database")

# ============================================
# 5. CREATE RAG APP CLASS FOR TRULENS
# ============================================
class RAGApp:
    """RAG Application wrapped for TruLens"""
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def retrieve(self, query: str):
        """Retrieve context for a query"""
        return retrieve_context(query, user_id="default_user_id", top_k=3)
    
    def generate(self, query: str, context: str):
        """Generate answer using context"""
        prompt = f"""Based on the following context, answer the question.
        
Context: {context}

Question: {query}

Answer:"""
        
        response = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def respond(self, query: str):
        """Full response pipeline - this is the main method TruLens will track"""
        context = self.retrieve(query)
        answer = self.generate(query, context)
        return answer, context

print("✅ RAG App class created!")

# ============================================
# 6. LOAD EVALUATION QUESTIONS
# ============================================
print("\n📚 Loading evaluation questions...")

eval_questions = []
file_loaded = False

try:
    with open('generated_questions.txt', 'r', encoding='utf-8') as file:
        for line in file:
            item = line.strip()
            if item:  # Skip empty lines
                eval_questions.append(item)
    file_loaded = True
    print(f"✅ Loaded {len(eval_questions)} evaluation questions from file")
except FileNotFoundError:
    print("⚠️ generated_questions.txt not found. Using sample questions.")
    eval_questions = [
        "What is an alphabet in automata theory?",
        "What is the symbol used to denote the empty string?",
        "What is the Kleene star operator (Σ*) used for?",
        "What is a formal language?",
        "Difference between ∅ and {ε}?"
    ]
    print(f"✅ Using {len(eval_questions)} sample questions")

# Show all questions
print(f"\n📋 Questions to evaluate ({len(eval_questions)} total):")
for i, q in enumerate(eval_questions, 1):
    print(f"  {i}. {q[:60]}...")

# ============================================
# 7. GET ANSWERS FOR ALL QUESTIONS
# ============================================
print("\n🔄 Getting answers from chatbot...")
answers = []
for i, question in enumerate(eval_questions, 1):
    try:
        print(f"  Processing {i}/{len(eval_questions)}: {question[:50]}...")
        response = get_gemini_response(question, user_id="eval_user")
        
        # Handle streaming response
        if hasattr(response, '__iter__') and not isinstance(response, str):
            full_response = ""
            for chunk in response:
                if chunk:
                    full_response += chunk
            answers.append(full_response)
        else:
            answers.append(str(response))
            
    except Exception as e:
        print(f"❌ Error: {question[:50]}... - {e}")
        answers.append(f"ERROR: {str(e)}")

print(f"✅ Completed {len(answers)}/{len(eval_questions)} questions")

# ============================================
# 8. RETRIEVE CONTEXTS FOR EVALUATION
# ============================================
print("\n🔄 Retrieving contexts for evaluation...")
contexts = []
for q in eval_questions:
    try:
        context = retrieve_context(q, user_id="default_user_id", top_k=3)
        contexts.append([context])
    except Exception as e:
        print(f"⚠️ Error retrieving context for '{q[:30]}...': {e}")
        contexts.append(["No context available"])
print("✅ Contexts retrieved")

# ============================================
# 9. FIX: IMPROVED METRIC PARSING
# ============================================
def parse_score(response_text: str) -> float:
    """Extract float score from response text"""
    try:
        # Try direct conversion first
        return float(response_text.strip())
    except:
        # Find all numbers in the response
        numbers = re.findall(r'(\d+\.?\d*)', response_text)
        if numbers:
            # Take the first number found
            return float(numbers[0])
        else:
            return 0.5

# ============================================
# 10. CREATE CUSTOM METRICS
# ============================================
client = Groq(api_key=GROQ_API_KEY)

def evaluate_context_relevance(input: str, context: list) -> float:
    """Evaluate if context is relevant to the question"""
    try:
        prompt = f"""Rate the relevance of the context to the question on a scale of 0-1.
        Return ONLY a number between 0 and 1.
        
        Question: {input}
        Context: {context[0][:500] if context and context[0] else "No context"}
        
        Relevance score (0-1):"""
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_score(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️ Error in context relevance: {e}")
        return 0.5

def evaluate_answer_relevance(input: str, output: str) -> float:
    """Evaluate if the answer is relevant to the question"""
    try:
        prompt = f"""Rate the relevance of the answer to the question on a scale of 0-1.
        Return ONLY a number between 0 and 1.
        
        Question: {input}
        Answer: {output[:500] if output else "No answer"}
        
        Relevance score (0-1):"""
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_score(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️ Error in answer relevance: {e}")
        return 0.5

def evaluate_groundedness(context: list, output: str) -> float:
    """Evaluate if the answer is grounded in the context"""
    try:
        prompt = f"""Rate how grounded the answer is in the context on a scale of 0-1.
        Return ONLY a number between 0 and 1.
        
        Context: {context[0][:500] if context and context[0] else "No context"}
        Answer: {output[:500] if output else "No answer"}
        
        Groundedness score (0-1):"""
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_score(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️ Error in groundedness: {e}")
        return 0.5

print("✅ Custom metrics created!")

# ============================================
# 11. SETUP TRULENS WITH TruApp
# ============================================
print("\n🔧 Setting up TruLens with TruApp...")

# Create Metrics
f_context_relevance = (
    Metric(evaluate_context_relevance)
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

f_answer_relevance = (
    Metric(evaluate_answer_relevance)
    .on_input()
    .on_output()
)

f_groundedness = (
    Metric(evaluate_groundedness)
    .on_context(collect_list=False)
    .on_output()
)

print("✅ Metrics defined!")

# ============================================
# 12. WRAP APP WITH TruApp
# ============================================
print("\n🚀 Wrapping app with TruApp...")

# Create your app instance
rag_app = RAGApp()

# Create variables to hold the recorder and results
tru_recorder = None
records_data = None

# Wrap with TruApp
if TRULENS_AVAILABLE:
    tru_recorder = TruApp(
        rag_app,
        app_name="RAG_Evaluation_App",
        app_version="v1.0",
        feedbacks=[f_context_relevance, f_answer_relevance, f_groundedness],
        main_method=rag_app.respond
    )
    print("✅ App wrapped with TruApp!")
else:
    print("⚠️ TruLens not available - running without recording")

# ============================================
# 13. RUN EVALUATION WITH TruApp
# ============================================
print("\n🚀 Running Evaluation...")
print(f"📊 Evaluating {len(eval_questions)} questions...")
print("=" * 60)

records = []
manual_records = []  # For manual tracking

if TRULENS_AVAILABLE and tru_recorder:
    # Use TruApp recorder context
    with tru_recorder as recording:
        for i, question in enumerate(eval_questions, 1):
            print(f"📝 Processing {i}/{len(eval_questions)}: {question[:50]}...")
            answer, context = rag_app.respond(question)
            print(f"  Answer preview: {answer[:100]}...")
            
            # Also store manually for verification
            manual_records.append({
                "question": question,
                "answer": answer,
                "context": context
            })
    
    print(f"\n✅ Evaluation recorded with TruLens! ({len(eval_questions)} questions)")
    
    # IMPORTANT: Wait for feedback computations to complete
    print("\n⏳ Waiting for feedback computations to complete...")
    time.sleep(5)  # Give feedback threads time to finish
    
    # Force garbage collection
    gc.collect()
    
else:
    # Run without TruLens recording
    for i, question in enumerate(eval_questions, 1):
        print(f"📝 Processing {i}/{len(eval_questions)}: {question[:50]}...")
        answer, context = rag_app.respond(question)
        print(f"  Answer preview: {answer[:100]}...")
        
        # Calculate scores manually
        context_score = evaluate_context_relevance(question, [context])
        answer_score = evaluate_answer_relevance(question, answer)
        grounded_score = evaluate_groundedness([context], answer)
        
        record = {
            "input": question,
            "output": answer,
            "context": context,
            "context_relevance": context_score,
            "answer_relevance": answer_score,
            "groundedness": grounded_score
        }
        records.append(record)

print("\n✅ Evaluation complete!")

# ============================================
# 14. FIX: GET RESULTS FROM TRULENS (Different method)
# ============================================
if TRULENS_AVAILABLE and tru_recorder:
    print("\n📊 Retrieving results from TruLens...")
    try:
        print("⏳ Waiting for feedbacks to finalize...")
        time.sleep(3)
        
        # Try different methods to get records
        records_data = None
        
        # Method 1: Try to get records from session
        try:
            if hasattr(session, 'get_records'):
                records_data = session.get_records()
            elif hasattr(session, 'get'):
                records_data = session.get()
            elif hasattr(session, 'records'):
                records_data = session.records
        except Exception as e:
            print(f"⚠️ Method 1 failed: {e}")
        
        # Method 2: Try to get from app
        if records_data is None and tru_recorder:
            try:
                if hasattr(tru_recorder, 'get_records'):
                    records_data = tru_recorder.get_records()
                elif hasattr(tru_recorder, 'records'):
                    records_data = tru_recorder.records
            except Exception as e:
                print(f"⚠️ Method 2 failed: {e}")
        
        # Method 3: Create DataFrame from manual records
        if records_data is None or len(records_data) == 0:
            print("⚠️ Could not retrieve from TruLens, creating manual results...")
            
            # Calculate scores for all questions manually
            df_manual = pd.DataFrame()
            inputs = []
            outputs = []
            context_scores = []
            answer_scores = []
            grounded_scores = []
            
            for i, q in enumerate(eval_questions):
                # Get the context and answer from manual_records
                if i < len(manual_records):
                    ctx = manual_records[i]['context']
                    ans = manual_records[i]['answer']
                else:
                    ctx = contexts[i][0] if i < len(contexts) and contexts[i] else "No context"
                    ans = answers[i] if i < len(answers) else "No answer"
                
                inputs.append(q)
                outputs.append(ans[:200] + "..." if len(ans) > 200 else ans)
                context_scores.append(evaluate_context_relevance(q, [ctx]))
                answer_scores.append(evaluate_answer_relevance(q, ans))
                grounded_scores.append(evaluate_groundedness([ctx], ans))
            
            df_manual['input'] = inputs
            df_manual['output'] = outputs
            df_manual['context_relevance'] = context_scores
            df_manual['answer_relevance'] = answer_scores
            df_manual['groundedness'] = grounded_scores
            
            records_data = df_manual
            print("✅ Created manual results DataFrame")
        
        # Display results
        if records_data is not None and len(records_data) > 0:
            # Convert to DataFrame if needed
            if not isinstance(records_data, pd.DataFrame):
                try:
                    df = records_data.to_pandas()
                except:
                    df = pd.DataFrame(records_data)
            else:
                df = records_data
            
            print("\n" + "=" * 60)
            print("📈 EVALUATION RESULTS SUMMARY")
            print("=" * 60)
            
            print(f"\n📊 Total Records: {len(df)}")
            print(f"📋 Expected Records: {len(eval_questions)}")
            
            if len(df) == len(eval_questions):
                print("✅ PERFECT! Records match expected count!")
            else:
                print(f"⚠️ Mismatch! Database has {len(df)} records, expected {len(eval_questions)}")
            
            # Display metrics
            print("\n📈 METRIC SCORES:")
            print("-" * 40)
            
            # Find metric columns
            metric_cols = []
            for col in df.columns:
                if any(metric in col.lower() for metric in ['context_relevance', 'answer_relevance', 'groundedness']):
                    metric_cols.append(col)
            
            if metric_cols:
                for col in metric_cols:
                    if col in df.columns:
                        mean_score = df[col].mean()
                        std_score = df[col].std()
                        min_score = df[col].min()
                        max_score = df[col].max()
                        
                        # Determine rating
                        if mean_score >= 0.8:
                            rating = "✅ Excellent"
                        elif mean_score >= 0.6:
                            rating = "✅ Good"
                        elif mean_score >= 0.4:
                            rating = "⚠️ Medium"
                        else:
                            rating = "❌ Poor"
                        
                        print(f"\n  {col.replace('_', ' ').title()}:")
                        print(f"    Mean:   {mean_score:.3f} {rating}")
                        print(f"    Std:    {std_score:.3f}")
                        print(f"    Min:    {min_score:.3f}")
                        print(f"    Max:    {max_score:.3f}")
            else:
                print("⚠️ No metric columns found in results")
                print(f"Available columns: {df.columns.tolist()}")
            
            # Show individual results
            print("\n📋 DETAILED RESULTS:")
            print("-" * 60)
            
            # Display first 10 results
            display_cols = []
            if 'input' in df.columns:
                display_cols.append('input')
            display_cols.extend([col for col in metric_cols if col in df.columns])
            
            if display_cols:
                for idx, row in df[display_cols].head(10).iterrows():
                    q = str(row.get('input', ''))[:50] + "..." if len(str(row.get('input', ''))) > 50 else str(row.get('input', ''))
                    print(f"\nQ{idx+1}: {q}")
                    for col in metric_cols:
                        if col in row:
                            print(f"  {col}: {row[col]:.3f}")
            else:
                print("No display columns found")
            
            # Save results
            df.to_csv("trulens_results.csv", index=False)
            print("\n💾 Results saved to trulens_results.csv")
            
        else:
            print("⚠️ No records found in TruLens database.")
            
    except Exception as e:
        print(f"⚠️ Error retrieving results: {e}")
        print("💡 Results may be available in the dashboard.")

# ============================================
# 15. SAVE MANUAL RESULTS (if TruLens not available)
# ============================================
if not TRULENS_AVAILABLE and records:
    df = pd.DataFrame(records)
    
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS (Manual)")
    print("=" * 60)
    
    print(f"\n📊 Total Records: {len(df)}")
    
    print("\n📈 METRIC SCORES:")
    print("-" * 40)
    
    for metric in ['context_relevance', 'answer_relevance', 'groundedness']:
        mean_score = df[metric].mean()
        if mean_score >= 0.8:
            rating = "✅ Excellent"
        elif mean_score >= 0.6:
            rating = "✅ Good"
        elif mean_score >= 0.4:
            rating = "⚠️ Medium"
        else:
            rating = "❌ Poor"
        
        print(f"\n  {metric.replace('_', ' ').title()}:")
        print(f"    Mean: {mean_score:.3f} {rating}")
        print(f"    Std:  {df[metric].std():.3f}")
        print(f"    Min:  {df[metric].min():.3f}")
        print(f"    Max:  {df[metric].max():.3f}")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 60)
    for i, row in df.iterrows():
        print(f"\nQ{i+1}: {row['input'][:50]}...")
        print(f"  Context Relevance: {row['context_relevance']:.3f}")
        print(f"  Answer Relevance:  {row['answer_relevance']:.3f}")
        print(f"  Groundedness:      {row['groundedness']:.3f}")
    
    # Save results
    df.to_csv("trulens_results.csv", index=False)
    print("\n💾 Results saved to trulens_results.csv")

# ============================================
# 16. FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)

print(f"\n✅ Questions Evaluated: {len(eval_questions)}")
print(f"✅ Total Records: {len(df) if 'df' in locals() and df is not None else 0}")

if TRULENS_AVAILABLE:
    print("✅ TruLens Dashboard: Available")
    print("🌐 Dashboard URL: http://localhost:8501 (or the port shown below)")
else:
    print("⚠️ TruLens Dashboard: Not Available")

# ============================================
# 17. CLEANUP
# ============================================
print("\n🧹 Cleaning up...")
gc.collect()

try:
    if TRULENS_AVAILABLE and tru_recorder:
        if hasattr(tru_recorder, 'cleanup'):
            tru_recorder.cleanup()
        elif hasattr(tru_recorder, '_cleanup'):
            tru_recorder._cleanup()
except Exception as e:
    print(f"⚠️ Cleanup warning: {e}")

# ============================================
# 18. LAUNCH DASHBOARD
# ============================================
print("\n🌐 Launching TruLens dashboard...")
time.sleep(2)

try:
    if TRULENS_AVAILABLE and tru_recorder:
        print("Opening dashboard...")
        # Use the new method
        from trulens.dashboard import run_dashboard
        run_dashboard()
    else:
        print("ℹ️ TruLens not available. Results saved to CSV.")
except Exception as e:
    print(f"⚠️ Dashboard error: {e}")
    print("📊 Results are available in trulens_results.csv")
    
    # Try alternative
    try:
        session.run_dashboard()
    except:
        pass

print("\n✅ Evaluation complete!")
print("=" * 60)

# ============================================
# 19. REGISTER CLEANUP ON EXIT
# ============================================
def cleanup_on_exit():
    """Cleanup function to run when script exits"""
    print("🧹 Running final cleanup...")
    gc.collect()

atexit.register(cleanup_on_exit)