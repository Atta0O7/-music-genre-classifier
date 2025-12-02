# cnn_mel_trainer.py
#
# Mel-spectrogram dataset (mel_data.npz) par 2D CNN train karta hai
# aur model + norm stats save karta hai.

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import json

DATA_PATH = "mel_data.npz"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "mel_cnn_model.keras"
NORM_STATS_PATH = MODELS_DIR / "mel_norm_stats.npz"
MAPPING_PATH = MODELS_DIR / "mel_genre_mapping.json"

print("\n📥 Loading mel dataset from:", DATA_PATH)
data = np.load(DATA_PATH)
X = data["X"]        # (N, 128,128,1)
y = data["y"]        # (N,)
mean = float(data["mean"])
std = float(data["std"])

print("X shape:", X.shape)
print("y shape:", y.shape)
num_classes = len(np.unique(y))
print("Number of classes:", num_classes)

# Normalize
X_norm = (X - mean) / (std + 1e-8)

# Train/val/test split (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
print("\n📂 Train:", X_train.shape, y_train.shape)
print("📂 Val:  ", X_val.shape, y_val.shape)
print("📂 Test: ", X_test.shape, y_test.shape)


def build_mel_cnn(input_shape, num_classes):
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.30))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.40))

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


input_shape = X_train.shape[1:]   # (128,128,1)
model = build_mel_cnn(input_shape, num_classes)
print("\n🧠 Model summary:")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
)

checkpoint_path = MODELS_DIR / "best_mel_cnn_weights.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)

EPOCHS = 40
BATCH_SIZE = 32

print("\n🚀 Starting Mel-CNN training...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1,
)

print("\n✅ Training complete!")

if checkpoint_path.exists():
    model.load_weights(checkpoint_path)
    print("✅ Best weights loaded from checkpoint.")

print("\n📏 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy (Mel-CNN): {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Save model
model.save(MODEL_PATH)
print("💾 Saved Mel-CNN model to:", MODEL_PATH)

# Save norm stats
np.savez(NORM_STATS_PATH, mean=mean, std=std)
print("💾 Saved norm stats to:", NORM_STATS_PATH)

print("\n🎉 Mel-CNN ready for Streamlit app.")
