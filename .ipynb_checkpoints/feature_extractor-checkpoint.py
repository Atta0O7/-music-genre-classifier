# feature_extractor.py
#
# Audio se numeric features nikalne ke liye helper functions.
# - extract_features(file_path)  -> 1D numpy array (length = 28)
# - build_features_csv(dataset_root, output_csv) -> features.csv banata hai

import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa


def extract_features(file_path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Ek audio file se 28 features nikalta hai.
    Yehi features tumhare CNN training ke liye use honge.

    Features:
    - 13 MFCC means
    - 1 spectral centroid mean
    - 1 spectral rolloff mean
    - 1 spectral bandwidth mean
    - 1 zero crossing rate mean
    - 1 RMS energy mean
    - 10 chroma STFT means
    ---------------------------------
    = 28 features total
    """

    # 1) Audio load karo (mono)
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # 2) MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = mfcc.mean(axis=1)  # shape (13,)

    # 3) Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    # 4) Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    # 5) RMS energy
    rms = librosa.feature.rms(y=y).mean()

    # 6) Chroma STFT (10 chroma bands)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=10)
    chroma_means = chroma.mean(axis=1)  # shape (10,)

    # Sab features ko ek array me jod do
    features = np.hstack(
        [
            mfcc_means,  # 13
            spectral_centroid,  # 1 -> 14
            spectral_rolloff,  # 1 -> 15
            spectral_bandwidth,  # 1 -> 16
            zcr,  # 1 -> 17
            rms,  # 1 -> 18
            chroma_means,  # 10 -> 28
        ]
    )

    # Ensure type float32 (model/scaler ke liye sahi)
    return features.astype(np.float32)


def build_features_csv(
    dataset_root: str,
    output_csv: str = "features.csv",
    audio_extensions=(".wav", ".mp3", ".au", ".ogg"),
) -> None:
    """
    Pura dataset directory se features nikal kar ek CSV banata hai.

    Expected folder structure (example):
        dataset_root/
            rock/
                file1.wav
                file2.wav
            jazz/
                file3.wav
            ...

    Har subfolder ka naam hi label/genre maanenge.
    CSV columns: f1 ... f28, label
    """

    dataset_root = Path(dataset_root)
    rows = []

    print("Scanning dataset root:", dataset_root)

    for genre_dir in sorted(dataset_root.iterdir()):
        if not genre_dir.is_dir():
            continue

        label = genre_dir.name  # folder name as genre/label
        print(f"Processing genre: {label}")

        for fname in sorted(os.listdir(genre_dir)):
            if not fname.lower().endswith(audio_extensions):
                continue

            file_path = genre_dir / fname

            try:
                feats = extract_features(str(file_path))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            row = {f"f{i+1}": float(feats[i]) for i in range(len(feats))}
            row["label"] = label
            rows.append(row)

    if not rows:
        print("No audio files found. Check dataset_root path.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved features for {len(rows)} files to {output_csv}")


if __name__ == "__main__":
    # Example usage:
    # Apne dataset ka root yaha set karo, jaise:
    # DATASET_ROOT = "database"
    # ya
    # DATASET_ROOT = "genres_original"
    DATASET_ROOT = "database"  # agar tumhara data "database" folder me hai

    build_features_csv(DATASET_ROOT, output_csv="features.csv")
