from flask import Flask, render_template, request, session
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = Flask(__name__)
app.secret_key = "secret123"
# ✅ Global storage for uploaded data
uploaded_df = None

# ✅ Load model & tokenizer
model = load_model("model/expense_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)


# ✅ Prediction Function
def predict_category(text):
    seq = tokenizer.texts_to_sequences([str(text)])
    padded = pad_sequences(seq, maxlen=10)
    pred = model.predict(padded)
    return labels[np.argmax(pred)]

# 🔥 Chatbot Intents
intents = {
    "total": ["total expenses", "overall spending", "how much spent"],
    "food": ["food expenses", "money spent on food", "food spending"],
    "travel": ["travel expenses", "money spent on travel"],
    "shopping": ["shopping expenses", "money spent on shopping"],
    "highest": ["highest category", "most spending category"],
    "average": ["average expense", "mean spending"]
}

# 🔥 Intent Matching Function
def get_intent(user_input):
    all_phrases = []
    intent_keys = []

    for key, phrases in intents.items():
        for p in phrases:
            all_phrases.append(p)
            intent_keys.append(key)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_phrases + [user_input])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()

    return intent_keys[index]

# ✅ Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        text = request.form.get("text")

        if text:
            result = predict_category(text)

    return render_template("index.html", result=result)


# ✅ CSV Upload Route
@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_df

    file = request.files["file"]

    # ✅ Handle encoding issue
    df = pd.read_csv(file, encoding="latin-1")

    # ✅ Normalize column names
    df.columns = [col.lower() for col in df.columns]

    # ✅ Detect text column automatically
    if "text" in df.columns:
        text_col = "text"
    else:
        text_col = df.columns[0]

    # ✅ Predict categories
    df["Predicted"] = df[text_col].apply(predict_category)

    # ✅ Save for chatbot
    uploaded_df = df

    # ✅ Chart logic (amount-based if available)
    if "amount" in df.columns:
        counts = df.groupby("Predicted")["amount"].sum()
    else:
        counts = df["Predicted"].value_counts()

    return render_template(
        "dashboard.html",
        tables=df.to_html(classes='data'),
        labels=list(counts.index),
        values=list(counts.values)
    )

import requests


def ask_ai(user_input, df):
    # Convert data to summary
    summary = ""

    if "amount" in df.columns:
        total = df["amount"].sum()
        summary += f"Total spending: ₹{total}\n"

        category_spend = df.groupby("Predicted")["amount"].sum()
        for cat, val in category_spend.items():
            summary += f"{cat}: ₹{val}\n"

    prompt = f"""
    You are a finance assistant.

    User data:
    {summary}

    User question:
    {user_input}

    Answer clearly.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# ✅ Chatbot Route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    global uploaded_df
    response = ""

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        user_input = request.form["message"]

        if uploaded_df is None:
            response = "Please upload a CSV file first."

        else:
            df = uploaded_df

            # 🔥 Use Gen AI
            response = ask_ai(user_input, df)

        session["history"].append({
            "user": user_input,
            "bot": response
        })

    return render_template("chatbot.html",
                           response=response,
                           history=session["history"])
# ✅ Run App
if __name__ == "__main__":
    app.run()