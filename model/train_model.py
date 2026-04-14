import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv("../data/data.csv", encoding="latin-1")

# Tokenization
tokenizer = Tokenizer(num_words=2000,oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=10)

# Labels
labels = list(df["label"].unique())
label_map = {l:i for i,l in enumerate(labels)}
y = np.array([label_map[l] for l in df["label"]])

# Model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64, input_length=5))
model.add(LSTM(64))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=50)

# Save model
model.save("model/expense_model.keras")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label map
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Model trained and saved!")
