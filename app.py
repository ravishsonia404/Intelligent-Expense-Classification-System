from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "expense_model.keras"

# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return "🚀 Expense Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction))

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
