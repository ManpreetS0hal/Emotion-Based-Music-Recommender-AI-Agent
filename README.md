# 🎵 Emotion-Based Music Recommender AI Agent

> Detects your emotion from a webcam image and recommends a matching Spotify playlist using a machine learning model and real-time AI interaction.

## 📌 Overview

This AI Agent captures your facial expression using your webcam, analyzes your emotion using a trained Convolutional Neural Network (CNN), and then recommends a Spotify playlist based on your mood.

Whether you're feeling happy, sad, or neutral, the system provides music that matches (or lifts) your emotion — all in real-time through a clean Streamlit interface.

## 🧠 Features

- 🎥 Real-time webcam emotion detection
- 🤖 CNN model trained on FER-2013 dataset
- 🎶 Spotify playlist suggestion using Spotipy API
- 🌐 Interactive UI built with Streamlit
- 🧩 Handles 7 emotions: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral

## 📊 How It Works

```mermaid
graph LR
A[Webcam Image] --> B[Emotion Prediction (CNN)]
B --> C[Emotion Label]
C --> D[Search Keyword]
D --> E[Spotify API (Spotipy)]
E --> F[Playlist Recommendation]
```

## 📂 Project Structure

```
emotion-music-recommender/
├── app.py                  # Streamlit application
├── my_model_62.keras       # Trained emotion detection model
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Optional UI settings
├── README.md               # Project description
```

## 📈 Model Details

- **Architecture**: CNN with 3 conv layers, max pooling, dropout  
- **Dataset**: [FER-2013 Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Accuracy**: ~70% validation accuracy  
- **Input Format**: 48x48 grayscale facial image

## 🤖 What Makes This an AI Agent?

| Component     | Role                                |
|---------------|-------------------------------------|
| Perception     | Webcam image via OpenCV             |
| Intelligence   | Emotion prediction via CNN          |
| Action         | Playlist recommendation via Spotify |
| Interaction    | Streamlit interface for users       |


## 🙋‍♂️ Author

**Manpreet Singh**    
[Doon University, Dehradun]
