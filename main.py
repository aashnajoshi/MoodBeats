"""
Install all required packages:
run 'pip install -r requirements.txt'

"""

import cv2  
import os
import re
import urllib.request  
import numpy as np
import player as pl
from keras.models import load_model

labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = load_model("model.h5")


def cam_not_accessed():
    user_feeling = input("How are you feeling today? ").strip().capitalize()

    if user_feeling in labels:
        emotion = user_feeling
        song_recommendations(emotion)
    else:
        print("Invalid emotion. Please choose from the following: ", labels)


def snapshot():
    cap = cv2.VideoCapture(0)
    name = "Snapshot.jpg"
    frames = []

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Can't access camera")
            cam_not_accessed()
        else:
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
                cv2.imwrite(name, frame)
                break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        cv2.imwrite(name, avg_frame)
        if "emotion" not in locals():
            emotion = emotion_from_camera(name)
        return emotion
    else:
        print("No frames captured.")
        cam_not_accessed()


def emotion_from_camera(name):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        faceROI = cv2.resize(faces[0], (48, 48), interpolation=cv2.INTER_NEAREST)
        faceROI = np.expand_dims(faceROI, axis=0)
        faceROI = np.expand_dims(faceROI, axis=3)
        prediction = model.predict(faceROI)
        return labels[int(np.argmax(prediction))]

    return "Unknown"


def song_recommendations(emotion):
    print(f"Detected emotion: {emotion}")
    csv_name = f"Song_Names/{emotion}.csv"
    if not os.path.exists(csv_name):
        print(f"No song recommendations found for emotion: {emotion}")
        cam_not_accessed()
    else:
        pl.play_random_song(csv_name)


def main():
    emotion = snapshot()

    if emotion is None:
        cam_not_accessed()

    song_recommendations(emotion)

def play_songs(emotion):
    song_list = pl.play_random_song(emotion)

    if not song_list:
        print("No song recommendations found.")
        return

    for song in song_list:
        url = "+".join(song.split())
        search_url = f"https://www.youtube.com/results?search_query={url}"
        try:
            html = urllib.request.urlopen(search_url)
            video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

            if video_ids:
                yt_link = f"https://www.youtube.com/watch?v={video_ids[0]}"
                print(f"Song: {song}, YouTube Link: {yt_link}")
                pl.play_random_song(yt_link)

        except Exception as e:
            print(f"Error in processing song {song}: {str(e)}")


if __name__ == "__main__":
    main()
