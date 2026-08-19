import os
import re
import math
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Safe optional imports for serverless environments
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except Exception as e:
    print("Pandas/NumPy import note:", e)
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception as e:
    print("Google GenerativeAI import note:", e)
    GENAI_AVAILABLE = False
    genai = None

# Pure NumPy TF-IDF Vectorizer
class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = []
        
    def fit_transform(self, raw_documents):
        docs = [str(doc).split() for doc in raw_documents]
        N = len(docs)
        if N == 0:
            return None
            
        vocab_set = {}
        for doc in docs:
            for word in set(doc):
                vocab_set[word] = vocab_set.get(word, 0) + 1
                
        self.vocab = {word: i for i, word in enumerate(vocab_set.keys())}
        self.idf = [math.log((N + 1) / (count + 1)) + 1 for count in vocab_set.values()]
        
        matrix = np.zeros((N, len(self.vocab)), dtype=np.float32)
        for d_idx, doc in enumerate(docs):
            if not doc:
                continue
            tf = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
            for word, count in tf.items():
                if word in self.vocab:
                    w_idx = self.vocab[word]
                    matrix[d_idx, w_idx] = (count / len(doc)) * self.idf[w_idx]
                    
            norm = np.linalg.norm(matrix[d_idx])
            if norm > 0:
                matrix[d_idx] = matrix[d_idx] / norm
                
        return matrix
        
    def transform(self, raw_documents):
        docs = [str(doc).split() for doc in raw_documents]
        matrix = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for d_idx, doc in enumerate(docs):
            if not doc:
                continue
            tf = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
            for word, count in tf.items():
                if word in self.vocab:
                    w_idx = self.vocab[word]
                    matrix[d_idx, w_idx] = (count / len(doc)) * self.idf[w_idx]
            norm = np.linalg.norm(matrix[d_idx])
            if norm > 0:
                matrix[d_idx] = matrix[d_idx] / norm
        return matrix

def simple_cosine_similarity(query_vec, doc_matrix):
    if query_vec is None or doc_matrix is None:
        return np.array([0.0])
    return np.dot(doc_matrix, query_vec.T).flatten()

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
    """Lazily load and vectorize dataset safely for serverless runtimes"""
    global _df, _vectorizer, _X
    if _df is not None:
        return _df, _vectorizer, _X

    if not PANDAS_AVAILABLE:
        _df = None
        _vectorizer = None
        _X = None
        return _df, _vectorizer, _X

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "updated_2.0_schemes.csv")

    try:
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
            _vectorizer = SimpleTfidfVectorizer()
            _X = _vectorizer.fit_transform(df_data["processed"])
        else:
            _vectorizer = None
            _X = None

        _df = df_data
    except Exception as e:
        print("Dataset loading note:", e)
        _df = None
        _vectorizer = None
        _X = None

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
        
    if not GENAI_AVAILABLE:
        return None

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
        return fallbacks[0]

    # 👉 Search dataset
    df, vectorizer, X = load_model_data()
    if df is None or df.empty or vectorizer is None or X is None:
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return fallbacks[0]

    try:
        q_vec = vectorizer.transform([query])
        scores = simple_cosine_similarity(q_vec, X)
        
        # Get top matches
        top_indices = np.argsort(scores)[::-1]
        idx = top_indices[0]
        best_score = scores[idx]

        # 👉 If match is extremely weak → use AI (threshold: 0.05)
        if best_score < 0.05:
            ai_response = get_ai_response(user_query)
            if ai_response:
                return ai_response
            return fallbacks[0]

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
            return fallbacks[0]

        # 👉 Limit response length
        sentences = ans.split(". ")
        ans = ". ".join(sentences[:6])
        if len(sentences) > 6:
            ans += "."

        return ans
    except Exception as e:
        print("Search calculation error:", e)
        ai_response = get_ai_response(user_query)
        if ai_response:
            return ai_response
        return fallbacks[0]