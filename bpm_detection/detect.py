import librosa
import warnings
import os
from math import ceil
from tqdm import tqdm
from glob import glob


def detect_bpm_single(file_name):
    """Detect BPM of an audio file."""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    samples, fs = librosa.load(file_name, sr=None, mono=True)
    bpm_array = librosa.beat.tempo(y=samples, sr=fs)
    bpm_v = bpm_array.tolist()[0]
    if bpm_v < 100:
        bpm_v = bpm_v * 2

    return ceil(bpm_v)

def detect_bpm(folder_name):
    """Detect BPM of all audio files in folder."""
    files = glob(folder_name + "/*.mp3")
    bpm_dict = {}
    for file in tqdm(files, desc=f"Detecting BPM in folder: {folder_name}"):
        filename = os.path.basename(file)
        bpm_dict[filename] = detect_bpm_single(file)
    return bpm_dict

if __name__ == "__main__":
    directory = "../asset"  # "your_audio_file.mp3"
    bpm = detect_bpm(directory)

    print(bpm)
