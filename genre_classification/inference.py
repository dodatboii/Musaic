import torch
import librosa
import librosa.feature as lf
import numpy as np
from pathlib import Path
from model import MusicCNN
import joblib
import os


class MusicGenrePredictor:
    def __init__(self, model_path, label_encoder_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Load the trained model
        self.checkpoint = torch.load(model_path, map_location=device)
        self.label_encoder = joblib.load(label_encoder_path)
        self.model = MusicCNN(num_genres=len(self.label_encoder.classes_))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Audio processing parameters (should match training)
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.target_frames = 1289

    def preprocess_audio(self, audio_path):
        """Preprocess a single audio file."""
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Extract MFCC features
        mfcc = lf.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)

        # Handle padding/trimming
        if mfcc.shape[1] < self.target_frames:
            padding = self.target_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        elif mfcc.shape[1] > self.target_frames:
            mfcc = mfcc[:, :self.target_frames]

        # Convert to tensor and add batch and channel dimensions
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor

    def predict_proba(self, audio_path):
        """Predict probability distribution over genres."""
        # Preprocess audio
        mfcc_tensor = self.preprocess_audio(audio_path)
        mfcc_tensor = mfcc_tensor.to(self.device)

        # Get model predictions
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            probabilities = torch.exp(output)  # Convert log_softmax to probabilities

        # Convert to numpy array
        probabilities = probabilities.cpu().numpy()[0]

        # Create dictionary of genre probabilities
        genre_probs = {
            genre: float(prob)
            for genre, prob in zip(self.label_encoder.classes_, probabilities)
        }

        # Sort by probability in descending order
        genre_probs = dict(sorted(genre_probs.items(), key=lambda x: x[1], reverse=True))
        return genre_probs

    def predict(self, audio_path):
        """Predict single genre label."""
        genre_probs = self.predict_proba(audio_path)
        predicted_genre = max(genre_probs.items(), key=lambda x: x[1])[0]
        return predicted_genre

    def predict_batch(self, audio_folder, extensions=None):
        """Predict genres for all audio files in a folder."""
        if extensions is None:
            extensions = ['.wav', '.mp3']
        results = {}
        audio_folder = Path(audio_folder)

        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_folder.glob(f'*{ext}'))

        # Process each file
        for audio_file in audio_files:
            try:
                genre_probs = self.predict_proba(str(audio_file))
                results[audio_file.name] = genre_probs
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

        return results


def print_prediction(file_name, genre_probs, top_k=3):
    """Helper function to print prediction results nicely."""
    print(f"\nPredictions for: {file_name}")
    print("-" * 50)
    print("Top {} genres:".format(top_k))
    for i, (genre, prob) in enumerate(list(genre_probs.items())[:top_k], 1):
        print(f"{i}. {genre}: {prob:.1%}")


def main(audio_path):
    # Initialize the predictor
    myPredictor = MusicGenrePredictor(
        model_path='src/best_model.pt',
        label_encoder_path='src/label_encoder.joblib'
    )

    if os.path.splitext(os.path.basename(audio_path))[1] in ['.wav', '.mp3']:
        # Get probability distribution for all genres
        genre_probs = myPredictor.predict_proba(audio_path)
        print_prediction(audio_path, genre_probs)

    else:
        results = myPredictor.predict_batch(audio_path)
        for file_name, genre_probs in results.items():
            genre_probs = myPredictor.predict_proba(audio_path)
            print_prediction(file_name, genre_probs)


if __name__ == '__main__':
    main(audio_path = "../asset")