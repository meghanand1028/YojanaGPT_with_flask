import os
import re
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Global cached dataset variables
_df = None
_vectorizer = None
_X = None

def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\u0900-\u097f\u0a80-\u0aff\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_data():
    """Lazily load and vectorize dataset to ensure instantaneous module imports"""
    global _df, _vectorizer, _X
    if _df is not None and _vectorizer is not None and _X is not None:
        return _df, _vectorizer, _X

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "updated_2.0_schemes.csv")

    if os.path.exists(CSV_PATH):
        df_data = pd.read_csv(CSV_PATH)
    elif os.path.exists("updated_2.0_schemes.csv"):
        df_data = pd.read_csv("updated_2.0_schemes.csv")
    else:
        df_data = pd.DataFrame()

    df_data = df_data.fillna("")
    text_cols = [
        'slug', 'details', 'benefits', 'eligibility',
        'application', 'documents', 'level',
        'schemeCategory', 'tags'
    ]
    valid_cols = [c for c in text_cols if c in df_data.columns]
    
    if valid_cols and not df_data.empty:
        df_data["combined"] = df_data[valid_cols].agg(" ".join, axis=1)
        df_data["processed"] = df_data["combined"].apply(normalize)
        _vectorizer = TfidfVectorizer()
        _X = _vectorizer.fit_transform(df_data["processed"])
    else:
        _vectorizer = TfidfVectorizer()
        _X = None

    _df = df_data
    return _df, _vectorizer, _X

# ---------------- INTENT KEYWORDS ----------------
benefit_words = ["benefit", "benefits", "फायदे", "लाभ", "फायदा", "advantage", "profit"]
doc_words = ["document", "documents", "कागदपत्र", "दस्तऐवज", "papers", "require", "required"]
elig_words = ["eligibility", "eligible", "पात्रता", "योग्यता", "qualify", "criteria", "age", "income"]
apply_words = ["apply", "application", "अर्ज", "process", "apply kaise", "how to apply", "registration", "register"]
scheme_words = ["yojana", "योजना", "scheme", "program", "ministry", "government", "subsidy", "pension", "ration"]

def is_scheme_query(query):
    q = query.lower()
    return any(word in q for word in scheme_words)

# ---------------- POLITE FALLBACKS ----------------
fallbacks = [
    "😊 Sorry, I couldn't find information about that.\nPlease ask me about government schemes like benefits, eligibility, documents or application process.",
    "🙏 I may not have data for this topic.\nTry asking about any government scheme and I’ll help you.",
    "🤖 I am trained mainly on government schemes.\nPlease ask something related to schemes, benefits, or documents."
]

# ---------------- AI MODEL INITIALIZATION ----------------
gemini_model = None

def get_gemini_model():
    global gemini_model
    if gemini_model is not None:
        return gemini_model
        
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            print("Gemini initialization error:", e)
    return gemini_model

def get_ai_response(user_query):
    """Call Google Gemini when dataset cannot answer"""
    model_inst = get_gemini_model()
    if not model_inst:
        return None

    try:
        response = model_inst.generate_content(
            f"You are a helpful assistant. Answer clearly and accurately in simple language. Question: {user_query}"
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)

    return None

# ---------------- RESPONSE FUNCTION ----------------
def get_response(user_query):
    query = normalize(user_query)

    # Greeting detection
    greetings = {"hi", "hello", "hey", "namaste", "नमस्कार"}
    if any(g in query.split() for g in greetings):
        return "Hello 👋 I can help you with government schemes. Ask me about benefits, eligibility, documents or application process."

    # 👉 If NOT scheme related → directly use AI
    if not is_scheme_query(query):
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return np.random.choice(fallbacks)

    # 👉 Search dataset
    df, vectorizer, X = load_model_data()
    if df.empty or X is None:
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return np.random.choice(fallbacks)

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X).flatten()
    
    # Get top matches
    top_indices = np.argsort(scores)[::-1]
    idx = top_indices[0]
    best_score = scores[idx]

    # 👉 If match is extremely weak → use AI (threshold: 0.05)
    if best_score < 0.05:
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return np.random.choice(fallbacks)

    row = df.iloc[idx]

    # 👉 Intent-based answer extraction
    ans = ""

    if any(w in query for w in benefit_words):
        ans = str(row.get("benefits", "")).strip()
    elif any(w in query for w in doc_words):
        ans = str(row.get("documents", "")).strip()
    elif any(w in query for w in elig_words):
        ans = str(row.get("eligibility", "")).strip()
    elif any(w in query for w in apply_words):
        ans = str(row.get("application", "")).strip()
    else:
        # For generic scheme name queries, check multiple fields in priority order
        ans = str(row.get("details", "")).strip()
        if not ans or len(ans) < 5:
            ans = str(row.get("benefits", "")).strip()
        if not ans or len(ans) < 5:
            ans = str(row.get("eligibility", "")).strip()
        if not ans or len(ans) < 5:
            ans = str(row.get("application", "")).strip()
        if not ans or len(ans) < 5:
            ans = str(row.get("schemeCategory", "")).strip()

    # 👉 If dataset answer empty → use AI
    if not ans or len(ans) < 5:
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return np.random.choice(fallbacks)

    # 👉 Limit response length
    sentences = ans.split(". ")
    ans = ". ".join(sentences[:6])
    if len(sentences) > 6:
        ans += "."

    return ans