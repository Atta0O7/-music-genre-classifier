# cnn_trainer.py  (updated, a bit stronger CNN + better monitoring)

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import joblib
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# CONFIG
# ==============================

CSV_PATH = "features.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "genre_cnn_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
MAPPING_PATH = MODELS_DIR / "genre_mapping.json"

# ==============================
# LOAD DATA
# ==============================

print("\n📥 Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded! Shape:", df.shape)

if "genre_label" not in df.columns:
    raise ValueError("Column 'genre_label' not found in features.csv")

label_col = "genre_label"
X = df.drop(columns=[label_col, "file_path"], errors="ignore").values
y = df[label_col].values

print("Features shape X:", X.shape)
print("Labels shape y:", y.shape)

# ==============================
# ENCODE LABELS
# ==============================

label_encoder = None

if np.issubdtype(y.dtype, np.number):
    print("\n🔢 Labels already numeric.")
    y_encoded = y.astype(int)
else:
    print("\n🔤 Encoding text labels with LabelEncoder...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
print("Number of classes:", num_classes)

print("\n📊 Class distribution (label -> count):")
unique_labels, counts = np.unique(y_encoded, return_counts=True)
for lbl, cnt in zip(unique_labels, counts):
    print(f"  Class {lbl}: {cnt} samples")

# ==============================
# TRAIN–TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n📂 Train shape:", X_train.shape, y_train.shape)
print("📂 Test shape:", X_test.shape, y_test.shape)

# ==============================
# SCALING
# ==============================

print("\n⚙️ Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]
print("Number of features:", n_features)

X_train_cnn = X_train_scaled.reshape(-1, n_features, 1)
X_test_cnn = X_test_scaled.reshape(-1, n_features, 1)

print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape:", X_test_cnn.shape)

# ==============================
# BUILD STRONGER CNN MODEL
# ==============================

def build_cnn_model(input_shape, num_classes):
    model = Sequential()

    # 1st Conv block
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    # 2nd Conv block
    model.add(Conv1D(128, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    # 3rd Conv block (new)
    model.add(Conv1D(256, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    # Dense head
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

input_shape = (n_features, 1)
model = build_cnn_model(input_shape, num_classes)
print("\n🧠 CNN model summary:")
model.summary()

# ==============================
# CLASS WEIGHTS (imbalance handle)
# ==============================

print("\n⚖️ Computing class weights...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_labels,
    y=y_encoded,
)
class_weights_dict = {int(lbl): float(w) for lbl, w in zip(unique_labels, class_weights)}
print("Class weights:", class_weights_dict)

# ==============================
# TRAIN
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
)

checkpoint_path = MODELS_DIR / "best_cnn_weights.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)

EPOCHS = 40
BATCH_SIZE = 32

print("\n🚀 Starting training...")

history = model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights_dict,
    verbose=1,
)

print("\n✅ Training complete!")

if checkpoint_path.exists():
    model.load_weights(checkpoint_path)
    print("✅ Best weights loaded from checkpoint.")

# ==============================
# EVALUATE
# ==============================

print("\n📏 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f" Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# ==============================
# SAVE MODEL + SCALER + MAPPING
# ==============================

print("\n💾 Saving model + scaler + mapping...")

model.save(MODEL_PATH)
print(" Saved CNN model to:", MODEL_PATH)

joblib.dump(scaler, SCALER_PATH)
print(" Saved scaler to:", SCALER_PATH)

if label_encoder is not None:
    classes = label_encoder.classes_
    genre_mapping = {int(i): str(label) for i, label in enumerate(classes)}
else:
    unique_labels_sorted = sorted(np.unique(y_encoded))
    genre_mapping = {int(i): int(i) for i in unique_labels_sorted}

with open(MAPPING_PATH, "w") as f:
    json.dump(genre_mapping, f)

print(" Saved genre mapping to:", MAPPING_PATH)
print("\n🎉 All done! Retrained CNN ready for Streamlit.")
