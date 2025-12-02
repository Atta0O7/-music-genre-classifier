import streamlit as st
import numpy as np
import tempfile
import pandas as pd
from pathlib import Path
import tensorflow as tf
import json
import librosa

# --------------------------------------------------
# 1. Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Music Genre Classifier (Mel-CNN)",
    page_icon="🎵",
    layout="wide"
)


# --------------------------------------------------
# 2. Background (same as before)
# --------------------------------------------------

def set_music_background():
    image_url = "https://images.unsplash.com/photo-1511379938547-c1f69419868d"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            animation: bg-pan 40s ease-in-out infinite alternate;
        }}

        @keyframes bg-pan {{
            0%   {{ background-position: 0% 50%; }}
            100% {{ background-position: 100% 50%; }}
        }}

        .main, .block-container {{
            background: rgba(0, 0, 0, 0.05) !important;
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border-radius: 20px;
            padding: 20px !important;
        }}

        section[data-testid="stSidebar"] > div {{
            background: rgba(0, 0, 0, 0.10) !important;
            backdrop-filter: blur(6px);
            border-radius: 20px;
        }}

        h1,h2,h3,h4,h5,p,span,label,div {{
            color: #ffffff !important;
            text-shadow: 0px 0px 8px black;
        }}

        .stSuccess {{
            background: rgba(0, 255, 120, 0.12) !important;
            border-left: 5px solid #22c55e !important;
        }}

        .stButton > button {{
            border-radius: 999px;
            background: rgba(0, 0, 0, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.4);
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.8rem;
            animation: glow 1.6s ease-in-out infinite alternate;
        }}

        @keyframes glow {{
            0%   {{ box-shadow: 0 0 10px rgba(255, 0, 150, 0.2); }}
            100% {{ box-shadow: 0 0 35px rgba(255, 0, 150, 0.9); }}
        }}

        .animated-title {{
            background: linear-gradient(90deg, #ff6ec4, #7873f5, #4ade80);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: title-anim 8s ease infinite alternate;
        }}

        @keyframes title-anim {{
            0%   {{ background-position: 0% 50%; }}
            100% {{ background-position: 100% 50%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_music_background()


# --------------------------------------------------
# 3. Model paths
# --------------------------------------------------

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mel_cnn_model.keras"
NORM_STATS_PATH = MODELS_DIR / "mel_norm_stats.npz"
MAPPING_PATH = MODELS_DIR / "mel_genre_mapping.json"

SR = 22050
DURATION = 3.0
SAMPLES_PER_CLIP = int(SR * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_FRAMES = 128


# --------------------------------------------------
# 4. Load model + norm stats + mapping
# --------------------------------------------------

@st.cache(allow_output_mutation=True)
def load_model_and_meta():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    stats = np.load(NORM_STATS_PATH)
    mean = float(stats["mean"])
    std = float(stats["std"])

    with open(MAPPING_PATH, "r") as f:
        genre_mapping = json.load(f)  # dict idx -> name

    # Ensure ordered list of genres
    num_classes = len(genre_mapping)
    genre_list = [genre_mapping[str(i)] if str(i) in genre_mapping else genre_mapping[i]
                  for i in range(num_classes)]

    return model, mean, std, genre_list


model, DATA_MEAN, DATA_STD, GENRE_NAMES = load_model_and_meta()


# --------------------------------------------------
# 5. Single audio -> mel spec (same as training)
# --------------------------------------------------

def audio_to_mel_spec(uploaded_file):
    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    y, sr = librosa.load(temp_path, sr=SR, mono=True)

    # center ka 3 sec segment
    if len(y) >= SAMPLES_PER_CLIP:
        start = (len(y) - SAMPLES_PER_CLIP) // 2
        y = y[start:start + SAMPLES_PER_CLIP]
    else:
        padding = SAMPLES_PER_CLIP - len(y)
        y = np.pad(y, (padding // 2, padding - padding // 2))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # crop/pad time axis
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

    mel_db = mel_db.astype(np.float32)

    # normalize with dataset mean/std
    mel_norm = (mel_db - DATA_MEAN) / (DATA_STD + 1e-8)

    # shape (1, 128,128,1)
    mel_norm = mel_norm[np.newaxis, ..., np.newaxis]

    return mel_norm


# --------------------------------------------------
# 6. Predict
# --------------------------------------------------

def predict_genre(uploaded_file):
    mel_input = audio_to_mel_spec(uploaded_file)  # (1,128,128,1)

    preds = model.predict(mel_input)
    probs = preds[0]  # (num_classes,)

    idx = int(np.argmax(probs))
    predicted_genre = GENRE_NAMES[idx]

    return predicted_genre, probs


# --------------------------------------------------
# 7. UI
# --------------------------------------------------

def main():
    st.markdown(
        """
        <h1 class="animated-title" style="text-align:center; font-size:48px; margin-top:-20px;">
            🎵 Mel-Spectrogram Music Genre Classifier
        </h1>
        <p style="text-align:center; font-size:20px;">
            Deep CNN using mel-spectrograms for 10-genre classification.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Upload WAV/MP3/OGG/AU",
            type=["wav", "mp3", "ogg", "au"]
        )

        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            st.info("Tip: 10–30 sec clear part of the song works best.")
            predict_btn = st.button("🎯 Predict Genre")
        else:
            predict_btn = False

    with col2:
        st.subheader("📊 Prediction & Probabilities")

        if uploaded_file and predict_btn:
            with st.spinner("Analyzing your audio with Mel-CNN... 🎧"):
                predicted_genre, probs = predict_genre(uploaded_file)

            df = pd.DataFrame({
                "Genre": GENRE_NAMES,
                "Probability": probs
            })
            df_sorted = df.sort_values("Probability", ascending=False)

            top3 = list(
                zip(
                    df_sorted["Genre"].head(3).tolist(),
                    df_sorted["Probability"].head(3).tolist()
                )
            )
            max_p = top3[0][1]

            if max_p > 0.70:
                confidence = "HIGH"
            elif max_p > 0.45:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            st.success(f"🎶 Predicted Genre: **{predicted_genre.upper()}**")
            st.write(
                f"Model confidence: **{confidence}** "
                f"(about {max_p*100:.1f}% sure on {top3[0][0].title()})"
            )

            st.markdown("**Top 3 possible genres:**")
            for name, p in top3:
                st.write(f"- {name.title()} — {p*100:.1f}%")

            with st.expander("See full probability table & bar chart (advanced)"):
                st.dataframe(df_sorted, use_container_width=True)
                st.bar_chart(
                    df_sorted.set_index("Genre")["Probability"],
                    use_container_width=True
                )

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
            Backend model: 2D CNN on **mel-spectrogram images**  
            Dataset: 10 genres, ~100 clips each (GTZAN-style).  
            Note: Still an educational project – real-world songs can be tricky!
        """)


if __name__ == "__main__":
    main()
