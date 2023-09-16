import cv2  # opencv-python
import os
import re
import urllib
from keras.models import load_model
import numpy as np
import pandas as pd
import player

MODEL_FILE = "model.h5"
SNAPSHOT_FILE = "Snapshot.jpg"
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def snapshot():
    cap = cv2.VideoCapture(0)
    ret = True
    frames = []

    while ret:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Can't access camera")
            break

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

        if cv2.waitKey(3) & 0xFF == ord("q"):
            cv2.imwrite(SNAPSHOT_FILE, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        cv2.imwrite(SNAPSHOT_FILE, avg_frame)
        emotion = emotion_from_camera(SNAPSHOT_FILE)
        os.remove(SNAPSHOT_FILE)
        return emotion
    else:
        print("No frames captured.")
        return None


def emotion_from_camera(snapshot_file):
    model = load_model(MODEL_FILE)
    face_cascade = cv2.CascadeClassifier(
        cv2.samples.findFile("haarcascade_frontalface_default.xml")
    )
    img = cv2.imread(snapshot_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for x, y, w, h in faces:
        faceROI = gray[y : y + h, x : x + w]
        faceROI = cv2.resize(faceROI, (48, 48), interpolation=cv2.INTER_NEAREST)
        faceROI = np.expand_dims(faceROI, axis=0)
        faceROI = np.expand_dims(faceROI, axis=3)
        prediction = model.predict(faceROI)
    return LABELS[int(np.argmax(prediction))]


def song_recommendations(emotion):
    print(f"Detected emotion: {emotion}")
    csv_name = f"Song_Names/{emotion}.csv"

    if not os.path.exists(csv_name):
        print(f"No song recommendations found for emotion: {emotion}")
        return []

    df = pd.read_csv(csv_name)
    data = df.values.tolist()
    r = random.sample(range(len(data)), 10)
    song_name = [str(data[i]).split("-")[0].strip("['") for i in r]
    return song_name


def main():
    emotion = snapshot()

    if emotion is None:
        return

    song_list = song_recommendations(emotion)

    if not song_list:
        print("No song recommendations found.")
        return

    for song in song_list:
        url = song.replace(" ", "+")
        search_url = "https://www.youtube.com/results?search_query=" + url
        try:
            html = urllib.request.urlopen(search_url)
            video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

            if video_ids:
                yt_link = "https://www.youtube.com/watch?v=" + video_ids[0]
                print(f"Song: {song}, YouTube Link: {yt_link}")
                player.play_random_song(emotion)

        except Exception as e:
            print(f"Error in processing song {song}: {str(e)}")


if __name__ == "__main__":
    main()
