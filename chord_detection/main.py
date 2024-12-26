import numpy as np
import os, sys, getopt
import matplotlib.pyplot as plt
import librosa
import json
from chromagram import compute_chroma
import hmm as hmm


def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord == "N":
            continue
        templates.append(templates_json[chord])

    return templates


def get_nested_circle_of_fifths():
    chords = [
        "N",
        "G",
        "G#",
        "A",
        "A#",
        "B",
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "Gm",
        "G#m",
        "Am",
        "A#m",
        "Bm",
        "Cm",
        "C#m",
        "Dm",
        "D#m",
        "Em",
        "Fm",
        "F#m",
    ]
    nested_cof = [
        "G",
        "Bm",
        "D",
        "F#m",
        "A",
        "C#m",
        "E",
        "G#m",
        "B",
        "D#m",
        "F#",
        "A#m",
        "C#",
        "Fm",
        "G#",
        "Cm",
        "D#",
        "Gm",
        "A#",
        "Dm",
        "F",
        "Am",
        "C",
        "Em",
    ]
    return chords, nested_cof


def find_chords(
    x: np.ndarray,
    fs: int,
    templates: list,
    chords: list,
    nested_cof: list = None,
    method: str = None,
    plot: bool = False,
):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords in it using 'method'
    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        templates: dictionary of chord templates
        chords: list of chords to search over
        nested_cof: nested circle of fifth chords
        method: template matching or HMM
        plot: if results should be plotted
    """

    # framing audio, window length = 8192, hop size = 1024 and computing PCP
    nfft = 8192
    hop_size = 1024
    nFrames = int(np.round(len(x) / (nfft - hop_size)))
    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    chroma = np.empty((num_chords // 2, nFrames))
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # step 1. compute chromagram
    for n in range(nFrames):
        xFrame[:, n] = x[start : start + nfft]
        start = start + nfft - hop_size
        timestamp[n] = n * (nfft - hop_size) / fs
        chroma[:, n] = compute_chroma(xFrame[:, n], fs)

    if method == "match_template":
        # correlate 12D chroma vector with each of
        # 24 major and minor chords
        for n in range(nFrames):
            cor_vec = np.zeros(num_chords)
            for ni in range(num_chords):
                cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
            max_cor[n] = np.max(cor_vec)
            id_chord[n] = np.argmax(cor_vec) + 1

        # if max_cor[n] < threshold, then no chord is played
        # might need to change threshold value
        id_chord[np.where(max_cor < 0.8 * np.max(max_cor))] = 0
        final_chords = [chords[cid] for cid in id_chord]

    elif method == "hmm":
        # get max probability path from Viterbi algorithm
        (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof)
        (path, states) = hmm.viterbi(PI, A, B)

        # normalize path
        for i in range(nFrames):
            path[:, i] /= sum(path[:, i])

        # choose most likely chord - with max value in 'path'
        final_chords = []
        indices = np.argmax(path, axis=0)
        final_states = np.zeros(nFrames)

        # find no chord zone
        set_zero = np.where(np.max(path, axis=0) < 0.3 * np.max(path))[0]
        if np.size(set_zero) > 0:
            indices[set_zero] = -1

        # identify chords
        for i in range(nFrames):
            if indices[i] == -1:
                final_chords.append("NC")
            else:
                final_states[i] = states[indices[i], i]
                final_chords.append(chords[int(final_states[i])])

    if plot:
        plt.figure()
        if method == "match_template":
            plt.yticks(np.arange(num_chords + 1), chords)
            plt.plot(timestamp, id_chord, marker="o")

        else:
            plt.yticks(np.arange(num_chords), chords)
            plt.plot(timestamp, np.int32(final_states), marker="o")

        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords


def main(input_file, method="match_template", plot=False):
    # Set default values for parameters
    has_method = True if method else False

    print("Input file is ", os.path.basename(input_file))
    print("Method is ", method)

    # Read the input file
    s, fs = librosa.load(path=input_file, duration=10)

    # Convert to mono if file is stereo
    x = s.tolist() if len(s.shape) else s

    # Get chords and circle of fifths
    chords, nested_cof = get_nested_circle_of_fifths()

    # Get chord templates
    templates = get_templates(chords)

    # Find the chords
    if method == "match_template":
        timestamp, final_chords = find_chords(
            x, fs, templates=templates, chords=chords, method=method, plot=plot
        )
    else:
        timestamp, final_chords = find_chords(
            x,
            fs,
            templates=templates,
            chords=chords[1:],  # excluding first chord
            nested_cof=nested_cof,
            method=method,
            plot=plot,
        )

    # Print chords with timestamps
    print("Time (s)", "Chord")
    for n in range(len(timestamp)):
        print("%.3f" % timestamp[n], final_chords[n])


if __name__ == "__main__":
    main(input_file=r"../asset/002.mp3", plot=True)
