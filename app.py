import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load emotion model
model = load_model("my_model_62.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]
))

emotion_music_map = {
    'Happy': 'Happy Vibes',
    'Sad': 'Soothing Songs',
    'Angry': 'Calm Music',
    'Surprise': 'Party Songs',
    'Fear': 'Relaxing Music',
    'Disgust': 'Instrumental Music',
    'Neutral': 'Chill Vibes'
}

emoji_map = {
    'Happy': '😄', 'Sad': '😢', 'Angry': '😠', 'Surprise': '😲',
    'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
}

def get_playlist_link(emotion):
    query = emotion_music_map.get(emotion, "Chill Vibes")
    result = sp.search(q=query, type='playlist', limit=1)
    items = result.get('playlists', {}).get('items', [])
    if items:
        return items[0]['external_urls']['spotify']
    return None

# Streamlit Page Setup
st.set_page_config(page_title="🎵 Moodify - Your Emotion Music Recommender", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f2f4f7;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #ff6a00, #ee0979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer {
        font-size: 14px;
        color: gray;
        margin-top: 40px;
        text-align: center;
    }
    .spotify-card {
        background-color: #1DB954;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# UI
st.markdown("<div class='title'>🎧 Moodify</div>", unsafe_allow_html=True)
st.subheader("Let your face pick your vibe.")
st.markdown("Upload a photo or use your webcam, and we'll detect your emotion to recommend a matching Spotify playlist!")

img_file = st.camera_input("Take a selfie to detect your emotion 👇")

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
    emoji = emoji_map.get(predicted_emotion, '')

    playlist_link = get_playlist_link(predicted_emotion)

    st.success(f"**Detected Emotion:** {predicted_emotion} {emoji}")

    if playlist_link:
        st.markdown(f"<a class='spotify-card' href='{playlist_link}' target='_blank'>🎵 Play {predicted_emotion} Playlist on Spotify</a>", unsafe_allow_html=True)
    else:
        st.warning("No playlist found. Try again!")

# Footer
st.markdown("<div class='footer'>Made with ❤️ by Manpreet Singh | </div>", unsafe_allow_html=True)
