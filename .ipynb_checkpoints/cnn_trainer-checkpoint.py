# cnn_trainer.py

# --- 1. Import necessary libraries for data handling ---

import pandas as pd
import numpy as np
from pathlib import Path

# For splitting data
from sklearn.model_selection import train_test_split

# For feature scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder

# For saving objects
import joblib
import json

# --- 2. Import TensorFlow / Keras and required layers for CNN ---

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==============================
# 3. CONFIG
# ==============================

CSV_PATH = "features.csv"   # tumhara features file
MODELS_DIR = Path("models")  # sab yahi save hoga
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "genre_cnn_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
MAPPING_PATH = MODELS_DIR / "genre_mapping.json"


# ==============================
# 4. LOAD DATA
# ==============================

print(" Loading dataset from:", CSV_PATH)

try:
    df = pd.read_csv(CSV_PATH)
    print(" Dataset loaded successfully!")
except Exception as e:
    print(f" Error loading dataset: {e}")
    raise SystemExit

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# yaha assume kar rahe hain ki last column label hai.
# Agar tumhare CSV me label ka naam 'label' ya 'genre' ho, to usko yaha set kar sakte ho:

if "label" in df.columns:
    label_col = "label"
elif "genre" in df.columns:
    label_col = "genre"
else:
    # agar last column label hai:
    label_col = df.columns[-1]

print("Using label column:", label_col)

X = df.drop(columns=[label_col]).values
y = df[label_col].values

print("Features shape X:", X.shape)
print("Labels shape y:", y.shape)


# ==============================
# 5. ENCODE LABELS (if needed)
# ==============================

label_encoder = None

if np.issubdtype(y.dtype, np.number):
    # already numeric labels
    print(" Labels are already numeric.")
    y_encoded = y.astype(int)
else:
    print(" Labels are text → encoding with LabelEncoder.")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
print("Number of classes:", num_classes)


# ==============================
# 6. TRAIN–TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# ==============================
# 7. SCALING FEATURES
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]
print("Number of features:", n_features)

# CNN ke liye reshape: (samples, timesteps, channels)
X_train_cnn = X_train_scaled.reshape(-1, n_features, 1)
X_test_cnn = X_test_scaled.reshape(-1, n_features, 1)

print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape:", X_test_cnn.shape)


# ==============================
# 8. BUILD CNN MODEL
# ==============================

def build_cnn_model(input_shape, num_classes):
    model = Sequential()

    # 1st Conv block
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # 2nd Conv block
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten + Dense
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


input_shape = (n_features, 1)
model = build_cnn_model(input_shape, num_classes)
model.summary()


# ==============================
# 9. TRAIN MODEL
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

checkpoint_path = MODELS_DIR / "best_cnn_weights.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)

EPOCHS = 30
BATCH_SIZE = 32

print(" Starting training...")

history = model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1,
)

print(" Training complete!")

# best weights load kar lo (optional but good)
if checkpoint_path.exists():
    model.load_weights(checkpoint_path)
    print(" Best weights loaded from checkpoint.")


# ==============================
# 🔍 10. EVALUATE MODEL
# ==============================

test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f" Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


# ==============================
#  11. SAVE MODEL, SCALER, MAPPING
# ==============================

# 1) CNN model
model.save(MODEL_PATH)
print(" Saved CNN model to:", MODEL_PATH)

# 2) Scaler (StandardScaler)
joblib.dump(scaler, SCALER_PATH)
print(" Saved scaler to:", SCALER_PATH)

# 3) Genre mapping
if label_encoder is not None:
    # labels text the, encode kiye the
    classes = label_encoder.classes_
    genre_mapping = {int(i): str(label) for i, label in enumerate(classes)}
else:
    # labels already numeric the; generic mapping bana dete hain
    unique_labels = sorted(np.unique(y_encoded))
    genre_mapping = {int(i): int(i) for i in unique_labels}

with open(MAPPING_PATH, "w") as f:
    json.dump(genre_mapping, f)

print(" Saved genre mapping to:", MAPPING_PATH)

print("\n All done! Model + scaler + mapping ready for Streamlit app.")
