import random

import torch
import torchaudio
from torchaudio import transforms


def open_audio_file(audio_file):
    """
    Load an audio file. Return the signal as a tensor and the sample rate
    """
    signal, sr = torchaudio.load(audio_file)
    return (signal, sr)


def rechannel_audio(audio, new_channel):
    """
    Change audio arr to drsired new_channel
    """
    signal, sr = audio
    if (signal.shape[0] == new_channel):
        # return as it is
        return audio

    if (new_channel == 1):
        # convert from stereo to mono by selecting only the first channel
        resignal = signal[:1, :]
    else:
        # convert from mono to stereo by duplicating the first channel
        resignal = torch.cat([signal, signal])
    return resignal, sr


def resample_audio(aud, new_sr):
    """
    Resample to desired new_sr to standardize sr between audio clips
    """
    sig, sr = aud
    if (sr == new_sr):
        # return as it is
        return aud

    num_channels = sig.shape[0]
    # resample first channel
    resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
    if (num_channels > 1):
        # resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
        resig = torch.cat([resig, retwo])

    return resig, new_sr


def pad_or_trunc_audio_to_len(aud, max_ms_len):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    """
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms_len

    if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:, :max_len]

    elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)


def time_shift_audio(audio, shift_limit):
    """
    Data Augmentation
    Shifts the signal to the left or right by some percent. Values at the end
    are 'wrapped around' to the start of the transformed signal.
    """
    signal, sr = audio
    _, sig_len = signal.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return signal.roll(shift_amt), sr


def get_mel_spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = audio
    top_db = 80
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    """
    Augment the Spectrogram by masking out some sections of it in both the frequency
    dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    overfitting and to help the model generalise better. The masked sections are
    replaced with the mean value.
    """
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(
            freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(
            time_mask_param)(aug_spec, mask_value)

    return aug_spec
