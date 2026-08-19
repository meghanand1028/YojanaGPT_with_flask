import pandas as pd
import numpy as np
import re
import os
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("updated_2.0_schemes.csv")
df = df.fillna("")

# ---------------- TEXT COLUMNS ----------------
text_cols = [
    'slug','details','benefits','eligibility',
    'application','documents','level',
    'schemeCategory','tags'
]

df["combined"] = df[text_cols].agg(" ".join, axis=1)

# ---------------- NORMALIZE TEXT ----------------
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\u0900-\u097f\u0a80-\u0aff\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["processed"] = df["combined"].apply(normalize)

# ---------------- TFIDF MODEL ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed"])

# ---------------- INTENT KEYWORDS ----------------
benefit_words = ["benefit","benefits","फायदे","लाभ","फायदा","advantage","profit"]
doc_words = ["document","documents","कागदपत्र","दस्तऐवज","papers","require","required"]
elig_words = ["eligibility","eligible","पात्रता","योग्यता","qualify","criteria","age","income"]
apply_words = ["apply","application","अर्ज","process","apply kaise","how to apply","registration","register"]
scheme_words = ["yojana","योजना","scheme","program","ministry","government","subsidy","pension","ration"]

# ---------------- SCHEME INTENT DETECTOR ----------------
def is_scheme_query(query):
    q = query.lower()
    return any(word in q for word in scheme_words)

# ---------------- POLITE FALLBACKS ----------------
fallbacks = [
    "😊 Sorry, I couldn't find information about that.\nPlease ask me about government schemes like benefits, eligibility, documents or application process.",
    "🙏 I may not have data for this topic.\nTry asking about any government scheme and I’ll help you.",
    "🤖 I am trained mainly on government schemes.\nPlease ask something related to schemes, benefits, or documents."
]

import google.generativeai as genai

# ---------------- API CONFIG ----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print("Gemini initialization error:", e)

def get_ai_response(user_query):
    """Call Google Gemini when dataset cannot answer"""
    if not gemini_model:
        return None

    try:
        response = gemini_model.generate_content(
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