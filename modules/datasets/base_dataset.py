import os
import os.path as osp
from typing import List, Dict

import numpy as np
import torch.utils.data as data
from modules.utils.audio_util import open_audio_file, rechannel_audio, resample_audio
from modules.utils.audio_util import get_mel_spectrogram, spectro_augment
from modules.utils.audio_util import pad_or_trunc_audio_to_len, time_shift_audio

AUDIO_EXTENSIONS = ['.wav']


def has_file_allowed_extension(filepath: str, extensions: List[str]):
    """check if a filepath has allowed extensions
    """
    filepath_lower = filepath.lower()
    return any(filepath_lower.endswith(ext) for ext in extensions)


def _find_classes(root_dir: str):
    """returns tuple class names & class to idx dicts from root_dir
    root_dir struct:
        root
            |_ class_x
                      |_ x1.ext
                      |_ x2.ext
            |_ class_y
                      |_ y1.ext
                      |_ y2.ext
    """
    # list of classes or subfolders under root_dir
    classes = [d for d in os.listdir(root_dir)
               if osp.isdir(osp.join(root_dir, d))]

    # class_name to index dict, order is based on os.listdir
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _make_dataset(dir: str, class_to_idx: Dict[str, int], extensions: List[str]):
    """returns a list of audio_fpath, class index tuples
    """
    audio_file_idx = []
    dir = osp.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = osp.join(dir, target)
        if not osp.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = osp.join(root, fname)
                    item = [path, class_to_idx[target]]
                    audio_file_idx.append(item)
    return audio_file_idx


def write_class_mapping_to_file(mapping, fpath) -> None:
    """mapping must be one level deep dict of class_names to index
    """
    with open(fpath, 'w') as fw:
        for class_name, class_idx in mapping.items():
            fw.write(''.join([class_name, ' ', str(class_idx), '\n']))


class BaseAudioDataset(data.Dataset):
    """A generic audio dataset
    Args:
        root (string): Root directory path.
        label_arr (list): arr of int labels.
        filepath_arr (list): arr of file paths to audio files.
        sampling_rate int: audio sampling rate, def:441000 hertz.
        channel int: audio channel, 1 for mono, 2 for stereo.
        duration int: audio sample duration in milli seconds.
        shift_pct float: time shift percentage for random audio shift.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(self,
                 label_arr,
                 filepath_arr,
                 sampling_rate=44100,
                 channel=2,
                 duration=4000,
                 shift_pct=0.4,
                 transform=None,
                 target_transform=None):
        self.label_arr = label_arr
        self.filepath_arr = filepath_arr
        self.sr = sampling_rate
        self.channel = channel
        self.duration = duration
        self.shift_pct = shift_pct
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filepath_arr)

    def __getitem__(self, idx):
        try:
            audio_file = self.filepath_arr[idx]
            class_id = self.label_arr[idx]

            audio = open_audio_file(audio_file)
            reaudio = resample_audio(audio, self.sr)
            rechan = rechannel_audio(reaudio, self.channel)

            # dur_aud = (num_channels, sr * time_in_ms, time_in_ms)
            dur_audio = pad_or_trunc_audio_to_len(rechan, self.duration)
            # shift the audio arr randomly to self.shift_pct
            shift_audio = time_shift_audio(dur_audio, self.shift_pct)
            # sgram = (num_channels, mel freq_bands, time_steps)
            sgram = get_mel_spectrogram(
                shift_audio, n_mels=64, n_fft=1024, hop_len=None)
            # sgram augmented with time & freq masks
            aug_sgram = spectro_augment(
                sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

            return aug_sgram, class_id  # X, y
        except Exception as e:
            print(e)
            return None

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of classes: {}\n'.format(
            len(np.unique(np.array(self.label_arr))))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class AudioFolderDataset(BaseAudioDataset):
    """An audio data loader where the audio files are arranged in this way:
        root
            |_ class_x
                      |_ x1.ext
                      |_ x2.ext
                      |_ x3.ext
            |_ class_y
                      |_ y1.ext
                      |_ y2.ext
                      |_ y3.ext
    Args:
        root (string): Root directory path.
        file_extensions (list[string]): A list of allowed extensions.
        transform (callable, optional):
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 root,
                 file_extensions=AUDIO_EXTENSIONS,
                 sampling_rate=44100,
                 channel=2,
                 duration=4000,
                 shift_pct=0.4,
                 transform=None,
                 target_transform=None):

        classes, class_to_idx = _find_classes(root)
        samples = _make_dataset(root, class_to_idx, file_extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root +
                               "\n" + "Supported extensions are: " +
                               ",".join(file_extensions)))
        filepath_arr = [fpath for fpath, _ in samples]
        label_arr = [label for _, label in samples]
        super(AudioFolderDataset, self).__init__(label_arr=label_arr,
                                                 filepath_arr=filepath_arr,
                                                 sampling_rate=sampling_rate,
                                                 channel=channel,
                                                 duration=duration,
                                                 shift_pct=shift_pct,
                                                 transform=transform,
                                                 target_transform=target_transform)


if __name__ == "__main__":
    train_data = AudioFolderDataset("data/UrbanSound8K/audio",
                                    file_extensions=AUDIO_EXTENSIONS)
    print("Printing AudioFolderDataset data class", train_data)
    print("Printing len of data class (Number of audio files in data)", len(train_data))
    print("Printing the shape of the first datum and its label from the data",
          train_data[0][0].shape, train_data[0][1])

    print()
