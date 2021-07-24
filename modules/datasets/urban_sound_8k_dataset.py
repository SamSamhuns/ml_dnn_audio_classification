"""
Audio Classifier Data loader
"""
import os.path as osp
from torch.utils.data import random_split
from modules.datasets import base_dataset
from modules.utils.util import read_csv


class UrbanSound8KDataset:
    def __init__(self,
                 data_root="data/UrbanSound8K/audio",
                 csv_annot_root="data/UrbanSound8K/metadata/UrbanSound8K.csv",
                 transform=None,
                 target_transform=None,
                 val_split=0.1,
                 test_split=0.1,
                 **kwargs):
        """
        data_root str: folder containing train & test data dirs
        csv_annot_root str: file path to annotation csv file

        transform torchaudio.transforms: transforms for train data
        val_split float: validation split fraction must be >= 0,
        test_split float: test split fraction must be >= 0,

        data_root
                |--fold1
                |--fold2
                |--fold3
                ...
        """
        # slice_file_name,fsID,start,end,salience,fold,classID,class
        data_list = read_csv(csv_annot_root)
        header = data_list[0]
        slice_fname_idx = header.index("slice_file_name")
        classID_idx = header.index("classID")
        fold_idx = header.index("fold")

        label_arr = []
        filepath_arr = []
        for data in data_list:
            slice_file_name = data[slice_fname_idx]
            classID = data[classID_idx]
            fold = data[fold_idx]
            label_arr.append(classID)
            filepath_arr.append(
                osp.join(data_root, f"fold{fold}", slice_file_name))
        full_dataset = base_dataset.BaseAudioDataset(label_arr,
                                                     filepath_arr,
                                                     transform=transform,
                                                     target_transform=target_transform)
        num_items = len(full_dataset)
        num_val = round(num_items * val_split)
        num_test = round(num_items * test_split)
        num_train = round(num_items) - num_val - num_test
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [num_train, num_val, num_test])

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds


if __name__ == "__main__":
    from modules.augmentations.audio_transforms import Compose
    from modules.augmentations.audio_transforms import ReSampleAudio, ReChannelAudio, PadOrTruncAudio
    from modules.augmentations.audio_transforms import TimeShiftAudio, MelSpectrogramAudio, AugmentSpectrum

    transform = Compose([
        ReSampleAudio(), ReChannelAudio(), PadOrTruncAudio(),
        TimeShiftAudio(), MelSpectrogramAudio(), AugmentSpectrum()
    ])
    UrbanSound8KData = UrbanSound8KDataset(data_root="data/UrbanSound8K/audio",
                                           csv_annot_root="data/UrbanSound8K/metadata/UrbanSound8K.csv",
                                           transform=transform)
    train_data = UrbanSound8KData.train_dataset
    val_data = UrbanSound8KData.val_dataset
    test_data = UrbanSound8KData.test_dataset
    print("Printing AudioFolderDataset train data class", train_data)
    print("Printing len of data class (Number of audio files in data)", len(train_data))
    print("Printing the shape of the first datum and its label from the data",
          train_data[0][0].shape, train_data[0][1])
    print("Printing AudioFolderDataset val data class", val_data)
    print("Printing len of data class (Number of audio files in data)", len(val_data))
    print("Printing the shape of the first datum and its label from the data",
          val_data[0][0].shape, val_data[0][1])
    print("Printing AudioFolderDataset test data class", test_data)
    print("Printing len of data class (Number of audio files in data)", len(test_data))
    print("Printing the shape of the first datum and its label from the data",
          test_data[0][0].shape, test_data[0][1])
