import csv
import random
from pytube import YouTube
import vlc
from youtubesearchpython import VideosSearch

CSV_FILE = "/Song_Name/{Emotion}.csv"


def load_songs(csv_file):
    """Load songs from a CSV file into a list of dictionaries."""
    with open(csv_file, mode="r", newline="") as file:
        return list(csv.DictReader(file))


def get_random_song(songs):
    """Choose a random song from the list of songs."""
    return random.choice(songs)


def search_youtube(song_name):
    """Search for a song on YouTube and return the video URL."""
    search_query = f"{song_name} official music video YouTube"
    print(f"Searching for '{song_name}' on YouTube...")

    videos_search = VideosSearch(search_query, limit=1)
    results = videos_search.result()
    return results["result"][0]["link"]


def play_song(video_url):
    """Play the audio of a YouTube video using VLC."""
    yt = YouTube(video_url)
    audio_stream = yt.streams.get_audio_only()
    player = vlc.MediaPlayer(audio_stream.url)
    player.play()
    return player


def main():
    songs = load_songs(CSV_FILE)

    while True:
        random_song = get_random_song(songs)
        song_name = random_song["Song Name"]
        video_url = search_youtube(song_name)
        player = play_song(video_url)

        user_input = input(
            "Press 'q' and Enter to quit, or Enter to play the next song: "
        ).strip()
        player.stop()
        if user_input.lower() == "q":
            break


if __name__ == "__main__":
    main()
