import cv2
import numpy as np
import csv
import random
import pywhatkit as pl
from keras.models import load_model
from youtubesearchpython import VideosSearch
import streamlit as st

# Load model and define labels
labels = ["Angry", "Romantic", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = load_model("model.h5")

def detect_emotion(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        faceROI = cv2.resize(
            img[
                faces[0][1] : faces[0][1] + faces[0][3],
                faces[0][0] : faces[0][0] + faces[0][2],
            ],
            (48, 48),
            interpolation=cv2.INTER_NEAREST,
        )
        faceROI = np.expand_dims(faceROI, axis=0)
        faceROI = np.expand_dims(faceROI, axis=3)
        prediction = model.predict(faceROI)
        return labels[int(np.argmax(prediction))]
    return None

def play_random_song(emotion):
    csv_name = f"Song_Names/{emotion}.csv"
    songs = []
    
    try:
        with open(csv_name, "r", encoding="utf-8") as file:
            songs = list(csv.DictReader(file))
    except FileNotFoundError:
        st.error(f"No song recommendations found for emotion: {emotion}")
        return
    
    if not songs:
        return
    
    played_songs = set()
    
    while True:
        remaining_songs = [
            song for song in songs if song["Song Name"] not in played_songs
        ]
        if not remaining_songs:
            st.warning("No more songs to play.")
            break

        random_song = random.choice(remaining_songs)
        song_name = random_song.get("Song Name")
        search_query = f"{song_name} official music video YouTube"
        videos_search = VideosSearch(search_query, limit=1)
        results = videos_search.result()

        if results["result"]:
            video_url = results["result"][0]["link"]
            try:
                st.success(f"Playing song: {song_name}, link {video_url}")
                pl.playonyt(video_url)
                played_songs.add(song_name)
                break  # Play one song at a time
            except Exception as e:
                st.error(f"Error in processing song {song_name}, link {video_url}: {str(e)}")
                break

# Streamlit app layout
st.title("Emotion-Based Song Player")

if st.button("Capture Emotion"):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Can't access camera")
        user_feeling = st.selectbox("How are you feeling today?", labels)
        play_random_song(user_feeling)
    else:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip horizontally

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, channels="RGB", caption="Captured Frame")
        
        emotion = detect_emotion(frame)
        if emotion:
            st.success(f"Detected emotion: {emotion}")
            play_random_song(emotion)
        else:
            st.warning("No face detected or emotion could not be determined.")
        
        cap.release()
