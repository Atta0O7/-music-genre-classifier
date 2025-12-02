# build_features_from_dataset.py
#
# Ye script GTZAN jaisa folder structure use karke
# 'features.csv' fir se banata hai using feature_extractor.extract_features

import os
from pathlib import Path
import pandas as pd
from feature_extractor import extract_features

# Yaha tumhara dataset ka path daalo
DATASET_DIR = Path("genres_original")  # e.g. genres_original/blues, genres_original/rock, etc.
OUTPUT_CSV = "features.csv"

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

all_rows = []

print("🔍 Scanning dataset:", DATASET_DIR.resolve())

for genre in GENRES:
    genre_dir = DATASET_DIR / genre
    if not genre_dir.exists():
        print(f"⚠️ Warning: Folder not found for genre: {genre_dir}")
        continue

    print(f"\n🎵 Processing genre: {genre}")
    for file_name in os.listdir(genre_dir):
        if not file_name.lower().endswith((".wav", ".au", ".mp3", ".ogg")):
            continue

        file_path = genre_dir / file_name

        try:
            feats = extract_features(str(file_path))  # shape (28,)
            row = {f"f{i}": feats[i] for i in range(len(feats))}
            row["genre_label"] = genre   # text label for now
            row["file_path"] = str(file_path)
            all_rows.append(row)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# DataFrame banaao
df = pd.DataFrame(all_rows)

# genre ko last me rakho, filename optional
cols = [c for c in df.columns if c not in ["genre_label", "file_path"]]
cols = cols + ["genre_label", "file_path"]
df = df[cols]

print("\n✅ Total samples extracted:", len(df))
print("Saving to:", OUTPUT_CSV)

df.to_csv(OUTPUT_CSV, index=False)
print("✅ features.csv successfully created!")
# build_features_from_dataset.py
#
# Ye script GTZAN jaisa folder structure use karke
# 'features.csv' fir se banata hai using feature_extractor.extract_features

import os
from pathlib import Path
import pandas as pd
from feature_extractor import extract_features

# Yaha tumhara dataset ka path daalo
DATASET_DIR = Path("genres_original")  # e.g. genres_original/blues, genres_original/rock, etc.
OUTPUT_CSV = "features.csv"

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

all_rows = []

print("🔍 Scanning dataset:", DATASET_DIR.resolve())

for genre in GENRES:
    genre_dir = DATASET_DIR / genre
    if not genre_dir.exists():
        print(f"⚠️ Warning: Folder not found for genre: {genre_dir}")
        continue

    print(f"\n🎵 Processing genre: {genre}")
    for file_name in os.listdir(genre_dir):
        if not file_name.lower().endswith((".wav", ".au", ".mp3", ".ogg")):
            continue

        file_path = genre_dir / file_name

        try:
            feats = extract_features(str(file_path))  # shape (28,)
            row = {f"f{i}": feats[i] for i in range(len(feats))}
            row["genre_label"] = genre   # text label for now
            row["file_path"] = str(file_path)
            all_rows.append(row)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# DataFrame banaao
df = pd.DataFrame(all_rows)

# genre ko last me rakho, filename optional
cols = [c for c in df.columns if c not in ["genre_label", "file_path"]]
cols = cols + ["genre_label", "file_path"]
df = df[cols]

print("\n✅ Total samples extracted:", len(df))
print("Saving to:", OUTPUT_CSV)

df.to_csv(OUTPUT_CSV, index=False)
print("✅ features.csv successfully created!")
