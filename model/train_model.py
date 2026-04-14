import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

print("🚀 Script started")

# Load dataset
df = pd.read_csv("data/data.csv", encoding="latin1")
print("✅ CSV Loaded")

# Clean column names
df.columns = df.columns.str.strip()
print("Columns:", df.columns)

# Check column
if "category" not in df.columns:
    print("❌ ERROR: 'category' column missing")
    exit()

# Encode category
# Encode category
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])

# 🔥 Convert text
if "text" in df.columns:
    df["text"] = df["text"].astype("category").cat.codes
    print("✅ Text encoded")

# Features & labels
X = df.drop("category", axis=1)
y = df["category"]

# Features
X = df.drop("category", axis=1)
y = df["category"]

print("X shape:", X.shape)

# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🔥 Starting training...")

# Train
model.fit(X, y, epochs=5)

print("💾 Saving model...")
model.save("expense_model.keras")

print("📂 Files now:", os.listdir())

print("✅ DONE")
