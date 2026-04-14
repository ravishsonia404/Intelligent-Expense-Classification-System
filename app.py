from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load model safely

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))


def predict(text, amount):
    text_vec = vectorizer.transform([text]).toarray()
    final_input = np.hstack((text_vec, [[amount]]))

    pred = model.predict(final_input)
    return le.inverse_transform(pred)[0]
# Categories (must match training)
categories = ["food", "travel", "shopping"]

# Home route
@app.route("/")
def home():
    return "🚀 Expense Classification API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data or "amount" not in data:
            return jsonify({"error": "Please provide 'amount' in JSON"}), 400

        amount = float(data["amount"])

        # Model prediction
        prediction = model.predict(np.array([[amount]]))
        predicted_class = int(np.argmax(prediction))
        category = categories[predicted_class]

        return jsonify({
            "amount": amount,
            "predicted_category": category
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
