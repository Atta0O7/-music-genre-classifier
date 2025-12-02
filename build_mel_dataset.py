# build_mel_dataset.py
#
# Audio files (genres_original/*/*.wav) ko mel-spectrogram images me convert karke
# ek .npz file me save karta hai: mel_data.npz

import os
from pathlib import Path
import numpy as np
import librosa
import json

# Dataset ka root folder (GTZAN style)
DATASET_DIR = Path("genres_original")   # yahi use kar rahe the
OUTPUT_NPZ = "mel_data.npz"
MAPPING_PATH = Path("models/mel_genre_mapping.json")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

SR = 22050           # sample rate
DURATION = 3.0       # har clip se 3 second use karenge
SAMPLES_PER_CLIP = int(SR * DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_FRAMES = 128   # final spectrogram size: (128 mel, 128 frames)

X_list = []
y_list = []

print("🔍 Scanning dataset:", DATASET_DIR.resolve())

for genre_idx, genre in enumerate(GENRES):
    genre_dir = DATASET_DIR / genre
    if not genre_dir.exists():
        print(f"⚠️ Folder not found for genre: {genre_dir}")
        continue

    print(f"\n🎵 Processing genre: {genre}")
    for file_name in os.listdir(genre_dir):
        if not file_name.lower().endswith((".wav", ".au", ".mp3", ".ogg")):
            continue

        file_path = genre_dir / file_name

        try:
            # 1) Audio load
            y, sr = librosa.load(file_path, sr=SR, mono=True)

            # 2) Center ka 3 sec segment lo
            if len(y) >= SAMPLES_PER_CLIP:
                start = (len(y) - SAMPLES_PER_CLIP) // 2
                y = y[start:start + SAMPLES_PER_CLIP]
            else:
                # short ho to pad
                padding = SAMPLES_PER_CLIP - len(y)
                y = np.pad(y, (padding // 2, padding - padding // 2))

            # 3) Mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                power=2.0,
            )  # shape (n_mels, time)

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # 4) Time dimension crop/pad to TARGET_TIME_FRAMES
            if mel_db.shape[1] >= TARGET_TIME_FRAMES:
                start = (mel_db.shape[1] - TARGET_TIME_FRAMES) // 2
                mel_db = mel_db[:, start:start + TARGET_TIME_FRAMES]
            else:
                pad_width = TARGET_TIME_FRAMES - mel_db.shape[1]
                mel_db = np.pad(
                    mel_db,
                    ((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=(mel_db.min(),),
                )

            # final shape: (128, 128)
            mel_db = mel_db.astype(np.float32)

            X_list.append(mel_db)
            y_list.append(genre_idx)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

X = np.stack(X_list, axis=0)   # (N, 128, 128)
y = np.array(y_list, dtype=np.int64)

# Add channel dimension for CNN: (N, 128, 128, 1)
X = X[..., np.newaxis]

print("\n✅ Done!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalization stats
mean = X.mean()
std = X.std()
print(f"\n📊 Dataset mean: {mean:.4f}, std: {std:.4f}")

# Save dataset + stats
np.savez(OUTPUT_NPZ, X=X, y=y, mean=mean, std=std)
print("💾 Saved mel dataset to:", OUTPUT_NPZ)

# Save genre mapping
genre_mapping = {int(i): g for i, g in enumerate(GENRES)}
with open(MAPPING_PATH, "w") as f:
    json.dump(genre_mapping, f)

print("💾 Saved genre mapping to:", MAPPING_PATH)
print("\n🎉 Mel dataset ready for CNN training.")
