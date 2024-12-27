import os
from pathlib import Path

from genre_classification.inference import MusicGenrePredictor
from bpm_detection.bpm_detect import detect_bpm_single
from chord_detection.chord_detect import run_chord_detect

def pipeline(file_name):
    print(f"Processing file: {os.path.basename(file_name)}")
    bpm = detect_bpm_single(file_name)
    print(f"BPM: {bpm}")
    genre_probs = myPredictor.predict_proba(file_name)
    print("\nTop 3 possible genres:")
    for i, (genre, prob) in enumerate(list(genre_probs.items())[:3], 1):
        print(f"{i}. {genre}: {prob:.1%}")
    print("\nChord flow:")
    run_chord_detect(input_file=file_name, method="match_template", bpm=bpm)


if __name__ == '__main__':
    obj = "asset"

    print("----------START----------")
    myPredictor = MusicGenrePredictor(
        model_path='genre_classification/src/best_model.pt',
        label_encoder_path='genre_classification/src/label_encoder.joblib'
    )

    if os.path.isdir(obj):
        audio_folder = Path(obj)
        audio_files = []
        for ext in ['.wav', '.mp3']:
            audio_files.extend(audio_folder.glob(f'*{ext}'))

        bpm_dict = {}
        for file in audio_files:
            try:
                pipeline(file)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
            print("-------------------------")

    elif os.path.isfile(obj):
        if not (obj.endswith('.wav') or obj.endswith('.mp3')):
            raise ValueError(f"The file {obj} is neither a .wav nor a .mp3 file.")
        pipeline(obj)

    else:
        raise ValueError(f"The path {obj} is neither a valid directory nor a valid file.")
