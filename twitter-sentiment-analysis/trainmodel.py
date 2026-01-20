import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Load Dataset (4 columns)
# =========================
df = pd.read_csv("twitter_training.csv", encoding="latin-1", header=None)

df.columns = ["tweet_id", "topic", "sentiment", "text"]

print("✅ Columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

# =========================
# Keep only text + sentiment
# =========================
df = df[['sentiment', 'text']].dropna()

# =========================
# Encode sentiment labels
# =========================
sentiment_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Irrelevant": 3
}

df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Drop unmapped rows (safety)
df = df.dropna()

# Use smaller sample for speed & memory safety
df = df.sample(20000, random_state=42)

# =========================
# Text Preprocessing
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# =========================
# Save Model & Vectorizer
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ model.pkl and vectorizer.pkl saved successfully")
