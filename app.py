import cv2
import numpy as np
import os
import re
import csv
import random
import streamlit as st
from tensorflow.keras.models import load_model
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model_path = os.path.join(os.getcwd(), "model.h5")
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def detect_emotion(frame):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face_roi = cv2.resize(img[y:y+h, x:x+w], (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=3)
        prediction = model.predict(face_roi, verbose=0)
        return labels[int(np.argmax(prediction))]
    except:
        return None

def clean_song_title(song_name):
    song_name = re.sub(r'[–—]', ' ', song_name)
    song_name = re.sub(r'\s*\([^)]+\)', '', song_name)
    song_name = re.sub(r'\bLYRICS\b', '', song_name, flags=re.IGNORECASE)
    return ' '.join(song_name.split())

def get_first_youtube_video_url(search_query):
    driver = None
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--log-level=3")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service(log_output=os.devnull)
        driver = webdriver.Edge(options=options, service=service)
        driver.get(f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}+official+music+video")
        videos = driver.find_elements(By.CSS_SELECTOR, 'a#video-title')
        for video in videos:
            title = video.get_attribute('title')
            link = video.get_attribute('href')
            if link and "watch?v=" in link and title and "official" in title.lower():
                return link
        for video in videos:
            link = video.get_attribute('href')
            if link and "watch?v=" in link:
                return link
        return None
    except:
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def play_random_song(emotion):
    csv_name = f"Song_Names/{emotion}.csv"
    try:
        with open(csv_name, "r", encoding="utf-8") as file:
            songs = [row[0].strip() for row in csv.reader(file) if row and row[0].strip()]
    except FileNotFoundError:
        st.error(f"No song recommendations found for emotion: {emotion}")
        return None, None
    except Exception as e:
        st.error(f"Error reading {csv_name}: {e}")
        return None, None

    if not songs:
        st.warning("No songs found in the CSV.")
        return None, None

    song = random.choice(songs)
    clean_name = clean_song_title(song)
    url = get_first_youtube_video_url(clean_name)
    return song, url

def cleanup():
    if 'driver' in st.session_state and st.session_state.driver:
        try:
            st.session_state.driver.quit()
        except:
            pass
        st.session_state.driver = None

st.title("MoodBeats: Your Mood, Your Music")

if 'emotion' not in st.session_state:
    st.session_state.emotion = None
if 'driver' not in st.session_state:
    st.session_state.driver = None
if 'song_played' not in st.session_state:
    st.session_state.song_played = False
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None
if 'current_song' not in st.session_state:
    st.session_state.current_song = None

try:
    if st.button("Capture Emotion"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Can't access camera.")
            st.session_state.emotion = st.selectbox("How are you feeling today?", labels)
        else:
            st.info("Focus the 'Webcam Feed' window and press 'q' to capture the frame.")
            cv2.namedWindow("Webcam Feed - Press 'q' to Capture")
            cv2.setWindowProperty("Webcam Feed - Press 'q' to Capture", cv2.WND_PROP_TOPMOST, 1)
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    st.session_state.emotion = st.selectbox("How are you feeling today?", labels)
                    break
                frame = cv2.flip(frame, 1)
                cv2.imshow("Webcam Feed - Press 'q' to Capture", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.session_state.captured_frame = frame
                    st.image(image_rgb, channels="RGB", caption="Captured Frame")
                    st.session_state.emotion = detect_emotion(frame)
                    if not st.session_state.emotion:
                        st.warning("No face detected or emotion could not be determined.")
                        st.session_state.emotion = st.selectbox("How are you feeling today?", labels)
                    else:
                        st.success(f"Detected emotion: {st.session_state.emotion}")
                    break
            cap.release()
            cv2.destroyAllWindows()

    if st.session_state.emotion:
        if st.button("Play"):
            if st.session_state.driver is None:
                try:
                    options = Options()
                    options.add_argument("--disable-gpu")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--log-level=3")
                    options.add_experimental_option('excludeSwitches', ['enable-logging'])
                    service = Service(log_output=os.devnull)
                    st.session_state.driver = webdriver.Edge(options=options, service=service)
                except Exception as e:
                    st.error(f"Failed to initialize browser: {e}")
                    st.session_state.driver = None
                    st.stop()

            song, url = play_random_song(st.session_state.emotion)
            if song and url:
                st.session_state.current_song = song
                try:
                    st.success(f"Playing: {song}")
                    st.session_state.driver.get(url + "&autoplay=1")
                    st.session_state.song_played = True
                except Exception as e:
                    st.error(f"Playback failed: {e}")
            elif song:
                st.warning(f"No direct link found. Opening search page.")
                try:
                    clean_name = clean_song_title(song)
                    st.session_state.driver.get(f"https://www.youtube.com/results?search_query={clean_name.replace(' ', '+')}+official+music+video&autoplay=1")
                    st.session_state.song_played = True
                except Exception as e:
                    st.error(f"Error opening search page: {e}")
finally:
    # if 'driver' in st.session_state:
    #     st.session_state.driver.quit()
    pass