import csv
import random
from pytube import YouTube
import vlc  # python-vlc
from youtubesearchpython import VideosSearch  # youtube-search-python
from main import snapshot as emotion

EMOTION = emotion
CSV_FILE = f"Song_Names/{EMOTION}.csv"
SEARCH_LIMIT = 5


def load_songs(csv_file):
    with open(csv_file, mode="r", newline="") as file:
        return list(csv.DictReader(file))


def search_youtube(song_name):
    search_query = f"{song_name} official music video YouTube"
    print(f"Searching for '{song_name}' on YouTube...")
    videos_search = VideosSearch(search_query, limit=SEARCH_LIMIT)
    results = videos_search.result()
    return [result["link"] for result in results["result"]] if results["result"] else []


def play_song(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.get_audio_only()
    player = vlc.MediaPlayer(audio_stream.url)
    player.play()
    return player


def main():
    songs = load_songs(CSV_FILE)

    while True:
        random_song = random.choice(songs)
        song_name = random_song["Song Name"]
        video_urls = search_youtube(song_name)

        if not video_urls:
            print(f"No results found for '{song_name}' on YouTube.")
            continue

        for video_url in video_urls:
            player = play_song(video_url)
            print(f"Playing {song_name}")
            user_input = input(
                "Press 'q' and Enter to quit, or Enter to play the next song: "
            ).strip()
            player.stop() 

            if user_input.lower() == "q":
                print("Goodbye!")
                break


if __name__ == "__main__":
    main()
