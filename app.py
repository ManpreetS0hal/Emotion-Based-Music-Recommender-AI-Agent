import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the trained emotion model
model = load_model("my_model_62.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify setup using Streamlit secrets
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]
))

# Emotion to keyword map
emotion_music_map = {
    'Happy': 'Happy Vibes',
    'Sad': 'Soothing Songs',
    'Angry': 'Calm Music',
    'Surprise': 'Party Songs',
    'Fear': 'Relaxing Music',
    'Disgust': 'Instrumental Music',
    'Neutral': 'Chill Vibes'
}

def get_playlist_link(emotion):
    query = emotion_music_map.get(emotion, "Chill Vibes")
    result = sp.search(q=query, type='playlist', limit=1)
    items = result.get('playlists', {}).get('items', [])
    if items:
        return items[0]['external_urls']['spotify']
    return "No playlist found"

# Streamlit UI
st.set_page_config(page_title="Emotion-Based Music Recommender", layout="centered")
st.title("🎵 Emotion-Based Music Recommender")
st.write("Capture your facial expression and get a playlist to match your mood!")

# Capture webcam image
img_file = st.camera_input("Take a photo")

if img_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(img_file.getvalue())
        temp_path = temp_file.name

    img = cv2.imread(temp_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)

    prediction = model.predict(reshaped)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    playlist_link = get_playlist_link(predicted_emotion)

    st.subheader(f"Detected Emotion: {predicted_emotion}")
    st.markdown(f"[🎧 Click here to listen to a playlist for {predicted_emotion}]({playlist_link})")
