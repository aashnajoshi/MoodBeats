import csv
import random
import vlc  
from pytube import YouTube
from youtubesearchpython import VideosSearch  
import sys


def load_songs(csv_file):
    with open(csv_file, mode="r", newline="") as file:
        return list(csv.DictReader(file))


def get_random_song(songs):
    return random.choice(songs)


def play_random_song(csv_file):
    songs = load_songs(csv_file)
    random_song = get_random_song(songs)
    song_name = random_song.get("Song Name")
    video_url = search_youtube(song_name)
    player = play_song(video_url)
    return player


def search_youtube(song_name):
    search_query = f"{song_name} Official Music Video"
    print(f"Searching for '{song_name}' on YouTube...")
    videos_search = VideosSearch(search_query, limit=1)
    results = videos_search.result()
    return results["result"][0]["link"]


def play_song(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.get_audio_only()
    player = vlc.MediaPlayer(audio_stream.url)
    player.play()
    return player


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py")
        sys.exit(1)

    emotion = sys.argv[1].strip().capitalize()
    while True:
        play_random_song(emotion)
