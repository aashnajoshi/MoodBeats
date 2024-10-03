# EmoTunes
EmoTunes is an emotion-based music recommendation system. It detects a user's emotion in real-time using a webcam, processes the image through a trained neural network model, and plays a YouTube song corresponding to the detected emotion. The songs are selected from CSV files based on different emotional categories (e.g., Happy, Sad). Users can also manually choose their emotions for a more customized experience.

## Features
- Real-time emotion detection via webcam
- Neural network model integration
- YouTube song playback according to detected emotions

## All required libraries can be installed using a single-line command:
```bash
pip install -r requirements.txt
```

## While to run the code:
### Console-based version:
```bash
python main.py
```
### Streamlit-based version:
```bash
streamlit run app.py
```
## Description about various files:
- `app.py:` Contains a streamlit-based version of the main code. 
- `main.py:` Core program logic for emotion detection and song playback.
- `model.h5:` Trained neural network model for facial emotion recognition.
- `Song_Names:` Folder containing CSV files with songs representing specific emotions.
- `requirements.txt:` File containing all required Python modules.
