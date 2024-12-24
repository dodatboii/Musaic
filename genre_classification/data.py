import os
import torch
from tqdm import tqdm
import librosa
import librosa.feature as lf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Custom Dataset class with preprocessing
class MusicDataset(Dataset):
    def __init__(self, data_path, sample_rate=22050, n_mfcc=13, target_frames=1289):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.target_frames = target_frames
        self.genres = os.listdir(data_path)
        self.data = []
        self.labels = []

        # Label encoding for genres
        self.label_encoder = LabelEncoder()

        # Load data and labels
        self._load_data()

    def _load_data(self):
        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            if os.path.isdir(genre_path):
                for file in tqdm(os.listdir(genre_path), desc=f"Loading {genre}"):
                    if file.endswith(".wav"):
                        try:
                            audio_path = os.path.join(genre_path, file)
                            # Extract features and store in the list
                            mfcc = self._extract_mfcc(audio_path)
                            self.data.append(mfcc)
                            self.labels.append(genre)
                        except Exception as e:
                            continue

        # Encode labels as integers
        self.labels = self.label_encoder.fit_transform(self.labels)

    def _extract_mfcc(self, audio_path):
        # Load audio using librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)

        # Pad or trim to a fixed length (target_frames)
        if mfcc.shape[1] < self.target_frames:
            # Pad with zeros if the sequence is shorter than target_frames
            padding = self.target_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        elif mfcc.shape[1] > self.target_frames:
            # Trim the sequence if it is longer than target_frames
            mfcc = mfcc[:, :self.target_frames]

        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the features and the corresponding label
        mfcc = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
