from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from enum import Enum
import librosa
import json
from chord_detection.chromagram import compute_chroma
import chord_detection.hmm as hmm


def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("chord_detection/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord == "N":
            continue
        templates.append(templates_json[chord])

    return templates


def get_nested_circle_of_fifths():
    chords = ["N","G","G#","A","A#","B","C","C#","D","D#","E","F","F#","Gm","G#m","Am","A#m","Bm","Cm","C#m","Dm","D#m","Em","Fm","F#m"]
    nested_cof = ["G","Bm","D","F#m","A","C#m","E","G#m","B","D#m","F#","A#m","C#","Fm","G#","Cm","D#","Gm","A#","Dm","F","Am","C","Em"]
    return chords, nested_cof


class ChordDetectionMethod(Enum):
    TEMPLATE_MATCHING = "template_matching"
    HMM = "hmm"


@dataclass
class ChordDetectionConfig:
    """Configuration parameters for chord detection"""
    fs: int  # sampling frequency (Hz)
    bpm: Optional[float] = None  # beats per minute
    beats_per_chord: int = 8  # number of beats per chord
    correlation_threshold: float = 0.8  # threshold for chord detection
    hmm_threshold: float = 0.3  # threshold for HMM no-chord detection


class ChordDetector:
    def __init__(self, config: ChordDetectionConfig, templates: List[np.ndarray],
                 chords: List[str], nested_cof: Optional[List] = None):
        """
        Initialize chord detector with given configuration and templates

        Args:
            config: ChordDetectionConfig object containing parameters
            templates: List of chord templates
            chords: List of chord names
            nested_cof: Nested circle of fifth chords for HMM
        """
        self.config = config
        self.templates = np.array(templates)
        self.chords = chords
        self.nested_cof = nested_cof
        self._initialize_frame_parameters()

    def _initialize_frame_parameters(self) -> None:
        """Calculate frame and hop sizes based on BPM if available"""
        if self.config.bpm is not None:
            beat_duration = 60 / self.config.bpm
            chord_duration = beat_duration * self.config.beats_per_chord
            samples_per_chord = int(chord_duration * self.config.fs)

            # Optimize frame size to nearest power of 2
            self.nfft = 2 ** int(np.ceil(np.log2(samples_per_chord)))
            self.hop_size = samples_per_chord
        else:
            self.nfft = 8192
            self.hop_size = 1024

    def _prepare_signal(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Prepare signal for processing and calculate number of frames"""
        # Calculate number of frames
        nFrames = int(np.round(len(x) / (self.nfft - self.hop_size)))

        # Zero pad signal if necessary
        padded_length = (nFrames * (self.nfft - self.hop_size)) + self.hop_size
        if len(x) < padded_length:
            x = np.pad(x, (0, padded_length - len(x)))

        return x, nFrames

    def _compute_chromagram(self, x: np.ndarray, nFrames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute chromagram from audio signal"""
        window = np.hanning(self.nfft)
        num_chords = len(self.templates)
        chroma = np.empty((num_chords // 2, nFrames))
        timestamp = np.zeros(nFrames)

        for n in range(nFrames):
            start = n * (self.nfft - self.hop_size)
            frame = x[start:start + self.nfft] * window
            timestamp[n] = start / self.config.fs
            chroma[:, n] = compute_chroma(frame, self.config.fs)

        # Apply temporal smoothing if using BPM
        if self.config.bpm is not None:
            frames_per_chord = int(self.hop_size / (self.nfft - self.hop_size))
            smoothing_kernel = np.ones(frames_per_chord) / frames_per_chord
            chroma = np.apply_along_axis(
                lambda x: np.convolve(x, smoothing_kernel, mode='same'),
                axis=1,
                arr=chroma
            )

        return chroma, timestamp

    def _template_matching(self, chroma: np.ndarray, nFrames: int) -> List[str]:
        """Detect chords using template matching method"""
        num_chords = len(self.templates)
        id_chord = np.zeros(nFrames, dtype=np.int32)
        max_cor = np.zeros(nFrames)

        # Vectorized correlation computation
        for n in range(nFrames):
            correlations = np.array([
                np.correlate(chroma[:, n], template)
                for template in self.templates
            ])
            max_cor[n] = np.max(correlations)
            id_chord[n] = np.argmax(correlations) + 1

        # Threshold calculation
        if self.config.bpm is not None:
            frames_per_chord = int(self.hop_size / (self.nfft - self.hop_size))
            thresh = np.convolve(
                max_cor,
                np.ones(frames_per_chord) / frames_per_chord,
                mode='same'
            ) * self.config.correlation_threshold
        else:
            thresh = self.config.correlation_threshold * np.max(max_cor)

        id_chord[max_cor < thresh] = 0
        return [self.chords[cid] for cid in id_chord]

    def _hmm_detection(self, chroma: np.ndarray, nFrames: int) -> List[str]:
        """Detect chords using HMM method"""
        if self.config.bpm is not None:
            frames_per_chord = int(self.hop_size / (self.nfft - self.hop_size))
            chord_frames = np.arange(0, nFrames, frames_per_chord)
            A = hmm.initialize_transitions(self.nested_cof, chord_frames, nFrames)
            PI, _, B = hmm.initialize(chroma, self.templates, self.chords, self.nested_cof)
        else:
            PI, A, B = hmm.initialize(chroma, self.templates, self.chords, self.nested_cof)

        path, states = hmm.viterbi(PI, A, B)
        path /= np.sum(path, axis=0)

        indices = np.argmax(path, axis=0)
        no_chord_mask = np.max(path, axis=0) < self.config.hmm_threshold * np.max(path)
        indices[no_chord_mask] = -1

        return [
            "NC" if idx == -1 else self.chords[int(states[idx, i])]
            for i, idx in enumerate(indices)
        ]

    def detect_chords(self, x: np.ndarray, method: ChordDetectionMethod) -> Tuple[np.ndarray, List[str]]:
        """
        Detect chords in audio signal using specified method

        Args:
            x: Input audio signal
            method: ChordDetectionMethod enum specifying detection method

        Returns:
            Tuple of (timestamps, detected chords)
        """
        if len(x.shape) > 1:
            raise ValueError("Input signal must be mono (1D)")

        x, nFrames = self._prepare_signal(x)
        chroma, timestamp = self._compute_chromagram(x, nFrames)

        if method == ChordDetectionMethod.TEMPLATE_MATCHING:
            final_chords = self._template_matching(chroma, nFrames)
        elif method == ChordDetectionMethod.HMM:
            final_chords = self._hmm_detection(chroma, nFrames)
        else:
            raise ValueError(f"Unsupported chord detection method: {method}")

        return timestamp, final_chords


def run_chord_detect(input_file, method="match_template", bpm = 100):
    audio_signal, fs = librosa.load(path=input_file)

    chords, nested_cof = get_nested_circle_of_fifths()
    templates = get_templates(chords)

    config = ChordDetectionConfig(
        fs=44100,
        bpm=bpm,
        beats_per_chord=8
    )

    detector = ChordDetector(config, templates, chords, nested_cof)

    if method == "match_template":
        timestamps, final_chords = detector.detect_chords(
            audio_signal,
            ChordDetectionMethod.TEMPLATE_MATCHING
        )
    else:
        timestamps, final_chords = detector.detect_chords(
            audio_signal,
            ChordDetectionMethod.HMM
        )

    print("->".join(final_chords[:8]))
    # print("Time (s)", "Chord")
    # for n in range(len(timestamp)):
    #     print("%.3f" % timestamp[n], final_chords[n])


if __name__ == "__main__":
    run_chord_detect(input_file=r"../asset/002.mp3", bpm=136)
