import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/genre_cnn_model.keras"
SCALER_PATH = "models/feature_scaler.pkl"
CSV_PATH = "features.csv"

print("📥 Loading model & scaler...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

print("📥 Loading features.csv...")
df = pd.read_csv(CSV_PATH)

if "genre_label" not in df.columns:
    raise ValueError("features.csv me 'genre_label' column nahi mila.")

X = df.drop(columns=["genre_label", "file_path"], errors="ignore").values
y = df["genre_label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_scaled = scaler.transform(X)
X_cnn = X_scaled.reshape(-1, X_scaled.shape[1], 1)

print("🔮 Predicting on full dataset...")
probs = model.predict(X_cnn)
y_pred_idx = np.argmax(probs, axis=1)

print("\n📊 Classification report:")
print(classification_report(y_encoded, y_pred_idx, target_names=le.classes_))

print("\n🧩 Confusion matrix:")
print(confusion_matrix(y_encoded, y_pred_idx))
