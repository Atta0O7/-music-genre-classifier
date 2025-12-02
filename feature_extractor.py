# feature_extractor.py
#
# Ye file training + Streamlit dono jagah SAME features degi
# Total features = 50
#
# Features:
# - ZCR (mean, std)
# - Spectral Centroid (mean, std)
# - Spectral Bandwidth (mean, std)
# - Spectral Rolloff (mean, std)
# - RMS Energy (mean, std)
# - 20 MFCCs (mean + std) = 40
#  => 10 + 40 = 50 features

import numpy as np
import librosa


def extract_features(file_path: str):
    """
    Ek audio file path leke 50-dimensional feature vector return karta hai.
    Yehi function:
      - features.csv banate time use hoga
      - Streamlit app me prediction ke time use hoga
    """

    # 1) Audio load karo (mono, fixed sample rate)
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # Agar audio bahut chhota ho to handle (kam se kam 1 sec)
    min_len = sr  # 1 second
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    # 2) Basic spectral features + unka mean & std

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_centroid_mean = float(np.mean(spec_centroid))
    spec_centroid_std = float(np.std(spec_centroid))

    # Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_bw_mean = float(np.mean(spec_bw))
    spec_bw_std = float(np.std(spec_bw))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # 3) MFCCs (20 coefficients → mean + std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # shape (20, time)
    mfcc_means = np.mean(mfcc, axis=1)                  # (20,)
    mfcc_stds = np.std(mfcc, axis=1)                    # (20,)

    # 4) Sabko ek vector me jodo
    features = np.hstack([
        zcr_mean, zcr_std,
        spec_centroid_mean, spec_centroid_std,
        spec_bw_mean, spec_bw_std,
        rolloff_mean, rolloff_std,
        rms_mean, rms_std,
        mfcc_means,
        mfcc_stds,
    ])

    # Safety: ensure 1D numpy array float32
    features = np.asarray(features, dtype=np.float32).flatten()

    return features
