# rf_trainer.py
#
# Random Forest based genre classifier.
# Ye features.csv use karega (jo feature_extractor se bana hai)
# aur ek pipeline (Scaler + RandomForest) train karke save karega.

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib

# -------------------------
# Paths
# -------------------------

CSV_PATH = "features.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RF_MODEL_PATH = MODELS_DIR / "rf_pipeline.joblib"


# -------------------------
# Load dataset
# -------------------------

print("\n📥 Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded! Shape:", df.shape)

if "genre_label" not in df.columns:
    raise ValueError("features.csv me 'genre_label' column nahi mila.")

X = df.drop(columns=["genre_label", "file_path"], errors="ignore").values
y = df["genre_label"].values

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------
# Encode labels
# -------------------------

print("\n🔤 Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

classes = le.classes_
print("Classes:", classes)

# -------------------------
# Train / test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n📂 Train shape:", X_train.shape, y_train.shape)
print("📂 Test shape:", X_test.shape, y_test.shape)

# -------------------------
# Build pipeline: Scaler + RF
# -------------------------

rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("rf", rf_clf),
    ]
)

# -------------------------
# Train
# -------------------------

print("\n🚀 Training Random Forest...")
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------

print("\n📏 Evaluating on test set...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy (Random Forest): {acc:.4f}")

print("\n📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n🧩 Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# Save model + label encoder
# -------------------------

print("\n💾 Saving RF pipeline + label encoder...")
obj = {
    "model": pipeline,
    "label_encoder": le,
}
joblib.dump(obj, RF_MODEL_PATH)

print(" Saved to:", RF_MODEL_PATH)
print("\n🎉 Random Forest model ready for Streamlit app.")
