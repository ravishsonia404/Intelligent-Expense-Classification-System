import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("🚀 Training started...")

# Load dataset
df = pd.read_csv("data/data.csv", encoding="latin1")

# Clean columns
df.columns = df.columns.str.strip()

# Check required columns
required_cols = ["text", "amount", "category"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Encode category
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])

# Text vectorization
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df["text"]).toarray()

# Combine features (text + amount)
X = np.hstack((text_features, df[["amount"]].values))
y = df["category"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Accuracy
accuracy = model.score(X, y)
print(f"✅ Accuracy: {accuracy:.2f}")

# Save everything
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("💾 Model & files saved successfully!")
