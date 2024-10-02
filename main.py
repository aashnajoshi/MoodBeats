import cv2
import numpy as np
import csv
import random
import pywhatkit as pl
from keras.models import load_model
from youtubesearchpython import VideosSearch

labels = ["Angry", "Romantic", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = load_model("model.h5")

def get_user_feeling():
    while True:
        user_feeling = input("How are you feeling today? ").strip().capitalize()

        if user_feeling in labels:
            return user_feeling
        else:
            print("Invalid emotion. Let's try again: ", labels)

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

def play_random_song(emotion):
    csv_name = f"Song_Names/{emotion}.csv"
    songs = []

    try:
        with open(csv_name, "r", encoding="utf-8") as file:
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
                    exit()
            except Exception as e:
                print(
                    f"Error in processing song {song_name}, link {video_url}: {str(e)}"
                )

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Can't access camera")
        user_feeling = get_user_feeling()
        play_random_song(user_feeling)
        return

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip horizontally
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite("snapshot.jpg", frame)
            print("Snapshot captured and saved as snapshot.jpg")
            emotion = detect_emotion(frame)
            if emotion:
                print(f"Detected emotion: {emotion}")
                play_random_song(emotion)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")