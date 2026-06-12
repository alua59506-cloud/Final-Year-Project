import re
import os
import joblib
import contractions
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# --- CONFIGURATION ---
# Use a relative path so it works on GitHub/Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'champion_model.pkl')

app = FastAPI(title="US Airline Sentiment API")

# --- NLP INITIALIZATION ---
# Pre-loading these for speed
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tk = TweetTokenizer(strip_handles=True, reduce_len=True)

# Logic from notebook: Keep negations, drop brand names [3]
negation_words = {'not', 'no', 'never', 'nor', 'cannot', 'hardly', 'barely', 'scarcely', 'seldom'}
custom_stopwords = stop_words - negation_words
airline_names = {'united', 'delta', 'southwest', 'americanair', 'usairways', 'virginamerica', 'american'}
custom_stopwords.update(airline_names)

champion_model = None

# --- MODELS ---
class TweetInput(BaseModel): # Fixed Syntax
    tweet: str

# --- UTILS ---
def clean_text(text: str):
    """Refined preprocessing matching the high-performing notebook version [4, 7]"""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(f'[^a-zA-Z!?\\s]', '', text)
    
    tokens = tk.tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word.isalpha() and word not in custom_stopwords
    ]
    return ' '.join(processed_tokens)

# --- API EVENTS & ENDPOINTS ---
@app.on_event("startup")
async def load_model():
    global champion_model
    try:
        # Load the SVM Pipeline (includes TF-IDF) [8]
        champion_model = joblib.load(MODEL_PATH)
        print(f"SUCCESS: Champion model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")

@app.post("/predict-sentiment/")
async def get_sentiment(tweet_input: TweetInput):
    if champion_model is None:
        return {"error": "Model not loaded on server."}

    # 1. Clean the input
    cleaned = clean_text(tweet_input.tweet)
    
    # 2. Predict (Pipeline handles vectorization internally [9])
    prediction = champion_model.predict([cleaned])
    
    return {
        "raw_tweet": tweet_input.tweet,
        "cleaned_tweet": cleaned,
        "predicted_sentiment": prediction
    }
