import gradio as gr
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# =========================
# Load Model & Vectorizer
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# Sentiment Label Mapping
# =========================
label_map = {
    0: "😡 Negative",
    1: "😐 Neutral",
    2: "😊 Positive",
    3: "🤷 Irrelevant"
}

# =========================
# Preprocessing Function
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# =========================
# Prediction Function
# =========================
def predict_sentiment(tweet):
    clean_tweet = preprocess_text(tweet)
    vectorized = vectorizer.transform([clean_tweet])

    # Prediction
    prediction = model.predict(vectorized)[0]

    # Confidence score
    probs = model.predict_proba(vectorized)[0]
    confidence = np.max(probs) * 100

    sentiment = label_map.get(prediction, "Unknown")

    return f"{sentiment}\n\nConfidence: {confidence:.2f}%"

# =========================
# Gradio Interface
# =========================
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a tweet here...",
        label="Tweet Text"
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Twitter Sentiment Analysis",
    description=(
        "Multi-class Twitter sentiment analysis using "
        "TF-IDF + Logistic Regression.\n\n"
        "Classes: Negative, Neutral, Positive, Irrelevant"
    ),
    theme="soft"
)

interface.launch()
