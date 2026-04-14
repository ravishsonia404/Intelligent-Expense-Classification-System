from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("expense_model.keras")

@app.route("/")
def home():
    return "🚀 Expense Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Example input: {"features": [1000, 1]}
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction))

    return jsonify({
        "prediction": predicted_class
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
