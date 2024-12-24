import random
import torch
import torchaudio
from torchaudio import transforms


class AudioUtil:

    @staticmethod
    def open(audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate

    @staticmethod
    def rechannel(aud, new_channel):
        waveform, sample_rate = aud

        if waveform.shape[0] == new_channel:
            return aud

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            new_waveform = waveform[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            new_waveform = torch.cat([waveform, waveform])

        return new_waveform, sample_rate

    @staticmethod
    def resample(aud, new_sample_rate):
        waveform, sample_rate = aud

        if sample_rate == new_sample_rate:
            return aud

        num_channels = waveform.shape[0]
        # Resample first channel
        new_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            new_second_channel = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[1:, :])
            new_waveform = torch.cat([new_waveform, new_second_channel])

        return new_waveform, new_sample_rate

    @staticmethod
    def pad_trunc(aud, max_ms):
        waveform, sample_rate = aud
        num_rows, sig_len = waveform.shape
        max_len = sample_rate // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            waveform = waveform[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            waveform = torch.cat((pad_begin, waveform, pad_end), 1)

        return waveform, sample_rate

    @staticmethod
    def time_shift(aud, shift_limit):
        waveform, sample_rate = aud
        _, sig_len = waveform.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return waveform.roll(shift_amt), sample_rate

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        waveform, sample_rate = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec