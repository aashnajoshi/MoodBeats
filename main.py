"""
Install all required packages:
run 'pip install -r requirements.txt'
"""

import cv2
import numpy as np
import csv
import random
import pywhatkit as pl
from keras.models import load_model
from youtubesearchpython import VideosSearch

labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = load_model("model.h5")


def not_com():
    user_feeling = input("How are you feeling today? ").strip().capitalize()

    if user_feeling in labels:
        emotion = user_feeling
        play_random_song(emotion)
        return True
    else:
        print("Invalid emotion. Please choose from the following: ", labels)
        return False


def detect_emotion():
    cap = cv2.VideoCapture(0)
    frames = []

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Can't access camera")
            not_com()

        frames.append(frame)
        cv2.putText(
            frame,
            "Press q to take snapshot",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        avg_frame = cv2.resize(avg_frame, (640, 480))  # Adjust image size
        cv2.imwrite("Snapshot.jpg", avg_frame)
        img = cv2.imread("Snapshot.jpg", cv2.IMREAD_GRAYSCALE)
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
        else:
            print("No frames captured.")
            not_com()


def play_random_song(emotion):
    csv_name = f"Song_Names/{emotion}.csv"
    songs = []

    try:
        with open(csv_name, mode="r", newline="") as file:
            songs = list(csv.DictReader(file))
    except FileNotFoundError:
        print(f"No song recommendations found for emotion: {emotion}")

    if not songs:
        return

    played_songs = set()

    while True:
        remaining_songs = [
            song for song in songs if song["Song Name"] not in played_songs
        ]
        if not remaining_songs:
            print("No more songs to play.")
            break

        random_song = random.choice(remaining_songs)
        song_name = random_song.get("Song Name")
        search_query = f"{song_name} official music video YouTube"
        videos_search = VideosSearch(search_query, limit=1)
        results = videos_search.result()

        if results["result"]:
            video_url = results["result"][0]["link"]
            try:
                print(f"Playing song: {song_name}, link {video_url}")
                pl.playonyt(video_url)
                played_songs.add(song_name)
                user_choice = (
                    input("Press 'Enter' to play another song, or 'x' to exit: ")
                    .strip()
                    .lower()
                )
                if user_choice == "x":
                    break
            except Exception as e:
                print(
                    f"Error in processing song {song_name}, link {video_url}: {str(e)}"
                )


def main():
    while True:
        if cv2.VideoCapture(0).isOpened():  # Check if the camera is accessible
            emotion = detect_emotion()
            if emotion:
                print(f"Detected emotion: {emotion}")
                play_random_song(emotion)
            break
        else:
            print("Can't access camera. Please check your camera.")
            not_com()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
