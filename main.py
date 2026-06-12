import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import contractions

# Ensure NLTK resources are downloaded (uncomment if running in a new environment)
# nltk.download(['stopwords', 'wordnet', 'punkt'], quiet=True)

# Initialize NLTK components once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tk = TweetTokenizer(strip_handles=True, reduce_len=True)

# RETAIN negation words (crucial for sentiment analysis)
negation_words = {'not', 'no', 'never', 'nor', 'cannot', 'hardly', 'barely', 'scarcely', 'seldom'}
custom_stopwords = stop_words - negation_words

# ADD airline names to stopwords
airline_names = {'united', 'delta', 'southwest', 'americanair', 'usairways', 'virginamerica', 'american'}
custom_stopwords.update(airline_names)

def clean_text(text):
    # Expand contractions
    text = contractions.fix(text)

    # Convert to lowercase
    text = text.lower()

    # Remove Twitter handles and URLs
    text = re.sub(r'@\w+', '', text) # Remove @mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs

    # Remove special characters, retaining '!' and '?'
    text = re.sub(f'[^a-zA-Z!?\\s]', '', text)

    # Tokenize text
    tokens = tk.tokenize(text)

    # Remove custom stopwords and lemmatize
    processed_tokens = []
    for word in tokens:
        if word.isalpha() and word not in custom_stopwords:
            processed_tokens.append(lemmatizer.lemmatize(word))

    return ' '.join(processed_tokens)

print("Preprocessing function `clean_text` defined and ready for use.")
def predict_sentiment(raw_tweet: str) -> str:
    """
    Predicts the sentiment of a raw tweet using the champion model.
    """
    if champion_model is None:
        return "Error: Model not loaded."

    # Step 1: Preprocess the raw tweet
    cleaned_tweet = clean_text(raw_tweet)

    # Step 2: Make prediction using the champion model
    # The champion model (pipeline) handles TF-IDF vectorization internally
    prediction = champion_model.predict([cleaned_tweet])

    return prediction[0]

# --- Example Usage / API Validation ---

# Sample test inputs
sample_inputs = [
    "great service loved the flight",
    "delayed again terrible experience",
    "flight was on time",
    "customer service is horrible!",
    "I am flying @SouthwestAir tomorrow. excited!"
]

print("\nTesting the `predict_sentiment` function with sample inputs:")
for i, tweet in enumerate(sample_inputs):
    predicted_sentiment = predict_sentiment(tweet)
    print(f"Input {i+1}: '{tweet}'\nPredicted Sentiment: '{predicted_sentiment}'\n")
  ```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
# Import joblib, os, clean_text, champion_model as defined above

app = FastAPI()

class TweetInput(BaseModel:
    tweet: str

@app.on_event("startup")
async def load_model():
    global champion_model
    model_path = os.path.join(PSENT, 'models', 'champion_model.pkl')
    try:
        champion_model = joblib.load(model_path)
        print("Champion model loaded for API service.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        champion_model = None

@app.post("/predict-sentiment/")
async def get_sentiment(tweet_input: TweetInput):
    if champion_model is None:
        return {"error": "Model not loaded, API not ready."}

    raw_tweet = tweet_input.tweet
    predicted_sentiment = predict_sentiment(raw_tweet) # Use the function defined above

    return {"tweet": raw_tweet, "predicted_sentiment": predicted_sentiment}

# To run this (after installing fastapi and uvicorn):
# uvicorn main:app --reload
```
