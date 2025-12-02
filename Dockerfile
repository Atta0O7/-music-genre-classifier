# Base image: minimal Linux + Python 3.9
FROM python:3.11-slim


# Working directory inside the container
WORKDIR /app

# Copy dependency file first (for layer caching)
COPY requirements.txt .

# Copy project files
COPY app.py .
COPY music_genre_cnn.h5 .
COPY scaler.joblib .

# Install dependencies (upgrade pip + no cache)
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Expose default Streamlit port
EXPOSE 8501

# Start Streamlit app when container runs
CMD ["streamlit", "run", "app.py"]
