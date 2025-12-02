# 🎵 Music Genre Classifier

A machine-learning based project that predicts the genre of an audio track using Mel-Spectrogram features and a Convolutional Neural Network (CNN).

---

## ⭐ Features
- Extracts Mel-Spectrogram features using Librosa  
- Trains a CNN for multi-class genre classification  
- Supports audio formats: WAV, MP3, OGG, AU  
- Clean and interactive Streamlit web interface  
- Shows prediction + confidence score  
- Lightweight and easy to run

---

## 📦 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Librosa**
- **NumPy / Pandas**
- **Scikit-learn**
- **Streamlit**

---

## 📂 Project Structure

music-genre-classifier/
│
├── app.py # Streamlit app
├── cnn_mel_trainer.py # CNN training script
├── feature_extractor.py # Mel feature extractor
├── build_mel_dataset.py # Dataset builder
├── models/ # Saved CNN model + scaler + mapping
├── requirements.txt # Dependencies
└── README.md # Documentation




## 🚀 Run the App

### 1. Install dependencies:
```bash
pip install -r requirements.txt
2. Start the Streamlit app:


streamlit run app.py
Upload an audio file to get the predicted genre.

🏋️ Train the Model
To retrain the CNN model:


python cnn_mel_trainer.py
This will save:

genre_cnn_model.keras

feature_scaler.pkl

mel_genre_mapping.json

Inside the models/ folder.

📈 Future Improvements
Higher accuracy with deeper CNN architectures

Data augmentation

Spectrogram visualizations

Online deployment

📜 License
Free to use for learning and research.


Made with ❤️ by Atta0O7
