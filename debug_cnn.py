# debug_cnn.py
# Ye script check karega ki CNN model har jagah ROCK bol raha hai ya nahi

import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/genre_cnn_model.keras"
SCALER_PATH = "models/feature_scaler.pkl"
CSV_PATH = "features.csv"

print("Loading model & scaler...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

print("Loading features.csv...")
df = pd.read_csv(CSV_PATH)

# X = saare features, y = genre_label
X = df.drop("genre_label", axis=1).values
y = df["genre_label"].values

print("Shape X:", X.shape, "y:", y.shape)

# Scaling same scaler se
X_scaled = scaler.transform(X)

# CNN input shape me reshape
X_cnn = X_scaled.reshape(-1, X_scaled.shape[1], 1)

# Sirf first 30 samples check karte hain
print("\nPredicting on first 30 samples from training data...")
probs = model.predict(X_cnn[:30])
pred_idx = np.argmax(probs, axis=1)

print("\nTrue labels (first 30):")
print(y[:30])

print("\nPredicted labels (first 30):")
print(pred_idx)

print("\nUnique predicted classes:")
print(np.unique(pred_idx))
