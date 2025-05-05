import cv2
import numpy as np
import os
import re
import csv
import random
import warnings
from tensorflow.keras.models import load_model
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
import logging

logging.disable(logging.CRITICAL)  # Disables all logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WDM_LOG'] = '0'  # Suppress WebDriver Manager logs
warnings.filterwarnings("ignore")
labels = ["Angry", "Romantic", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

try:
    model = load_model("model.h5")
except:
    exit()

def get_user_feeling():
    while True:
        user_feeling = input("Your Cam is busy... How are you feeling? ").strip().capitalize()
        if user_feeling in labels:
            return user_feeling
        print(f"Invalid emotion. Try one of: {labels}")

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
        emotion = labels[int(np.argmax(prediction))]
        return emotion
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
    except:
        print(f"Error: Could not read {csv_name}")
        return
    if not songs:
        print("No songs found in the CSV")
        return

    driver = None
    try:
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--log-level=3")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service(log_output=os.devnull)
        driver = webdriver.Edge(options=options, service=service)
        while True:
            song = random.choice(songs)
            clean_name = clean_song_title(song)
            url = get_first_youtube_video_url(clean_name)
            print(f"Playing: {song}")
            if url:
                driver.get(url + "&autoplay=1")
            else:
                driver.get(f"https://www.youtube.com/results?search_query={clean_name.replace(' ', '+')}+official+music+video&autoplay=1")
            user_choice = input("Press 'Enter' to play another song, or 'x' to exit: ").strip().lower()
            if user_choice == 'x':
                break
    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    emotion = None
    if not cap.isOpened():
        emotion = get_user_feeling()
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    emotion = detect_emotion(frame)
                    if not emotion:
                        emotion = get_user_feeling()
                    break
        except:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    if emotion:
        os.system('cls' if os.name == 'nt' else 'clear') 
        print(f"Emotion Detected: {emotion}")
        play_random_song(emotion)

if __name__ == "__main__":
    main()