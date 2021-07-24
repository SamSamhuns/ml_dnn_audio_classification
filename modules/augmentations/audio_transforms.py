import random
import torch
import torchaudio


class Compose:
    """Composes several transforms together for audio.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.ReSampleAudio()
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ReSampleAudio(object):
    """
    Resample to desired new_sr to standardize sr between audio clips
    """

    def __init__(self, target_sr=44100):
        self.target_sr = target_sr

    def __call__(self, audio):
        signal, sr = audio
        if (sr == self.target_sr):
            # return as it is
            return audio

        num_channels = signal.shape[0]
        # resample first channel
        resig = torchaudio.transforms.Resample(
            sr, self.target_sr)(signal[:1, :])
        if (num_channels > 1):
            # resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(
                sr, self.target_sr)(signal[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, self.target_sr


class ReChannelAudio(object):
    """ReChannel audio sample to target_num_channel.
    Args:
        target_num_channel (int): Desired target num_channels
    """

    def __init__(self, target_num_channel=2):
        assert isinstance(target_num_channel, int)
        self.target_num_channel = target_num_channel

    def __call__(self, audio):
        signal, sr = audio
        if (signal.shape[0] == self.target_num_channel):
            # return as it is
            return audio
        if (self.target_num_channel == 1):
            # convert from stereo to mono by selecting only the first channel
            resignal = signal[:1, :]
        else:
            # convert from mono to stereo by duplicating the first channel
            resignal = torch.cat([signal, signal])
        return resignal, sr


class PadOrTruncAudio(object):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    """

    def __init__(self, max_ms_len=4000):
        self.max_ms_len = max_ms_len

    def __call__(self, audio):
        signal, sr = audio
        num_rows, sig_len = signal.shape
        max_len = sr // 1000 * self.max_ms_len

        if (sig_len > max_len):
            # Truncate the signal to the given length
            signal = signal[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            signal = torch.cat((pad_begin, signal, pad_end), 1)

        return signal, sr


class TimeShiftAudio(object):
    """
    Shifts the signal to the left or right by some percent. Values at the end
    are 'wrapped around' to the start of the transformed signal.
    """

    def __init__(self, shift_pct=0.4):
        self.shift_pct = shift_pct

    def __call__(self, audio):
        signal, sr = audio
        _, sig_len = signal.shape
        shift_amt = int(random.random() * self.shift_pct * sig_len)
        return signal.roll(shift_amt), sr


class MelSpectrogramAudio(object):
    """
    Get Mel Spectrogram from audio sample
    """

    def __init__(self, n_mels=64, n_fft=1024, hop_len=None):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len

    def __call__(self, audio):
        signal, sr = audio
        top_db = 80
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=self.n_fft, hop_length=self.hop_len, n_mels=self.n_mels)(signal)

        # convert to decibels/log scale
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec


class AugmentSpectrum(object):
    """
    Augment the Spectrogram by masking out some sections of it in both the frequency
    dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    overfitting and to help the model generalise better. The masked sections are
    replaced with the mean value.
    """

    def __init__(self, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        self.max_mask_pct = max_mask_pct
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spec):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = self.max_mask_pct * n_mels
        for _ in range(self.n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(
                freq_mask_param)(aug_spec, mask_value)

        time_mask_param = self.max_mask_pct * n_steps
        for _ in range(self.n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(
                time_mask_param)(aug_spec, mask_value)

        return aug_spec


class ToTensor(object):
    """Convert an data to tensor.
    This transform does not support torchscript.
    """

    def __call__(self, audio):
        """
        Args:
            data: data to be converted to tensor.

        Returns:
            Tensor: Converted data sample.
        """
        return torch.as_tensor(int(audio))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """
    Normalize tensor object
    """

    def __call__(self, data):
        """
        Args:
            data: data to be normalized.

        Returns:
            Tensor: Normalized tensor data.
        """
        mean_norm = (data - data.mean()) / data.std()
        return torch.nan_to_num(mean_norm)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Preprocess:
    common_transform = Compose([
        ReSampleAudio(), ReChannelAudio(), PadOrTruncAudio(),
        TimeShiftAudio(), MelSpectrogramAudio(), AugmentSpectrum(), Normalize()
    ])
    train = common_transform
    val = common_transform
    test = common_transform
    inference = common_transform
    target = Compose([ToTensor()])

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        pass
