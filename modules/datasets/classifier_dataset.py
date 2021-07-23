"""
Audio Classifier Data loader
"""
import os.path as osp

from modules.datasets import base_dataset


class ClassifierDataset:
    def __init__(self,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 data_mode="audio",
                 data_root='data',
                 train_dir='train',
                 val_dir='val',
                 test_dir='test',
                 **kwargs):
        """
        train_transform: torchvision.transforms for train data
        val_transform: torchvision.transforms for validation data
        test_transform: torchvision.transforms for test data
        data_mode: Mode for getting data
            data_root: folder containing train & test data dirs
            train_dir: train dir under data_root
            val_dir: val dir under data_root
            test_dir: test dir under data_root

        data_root
                |--train_dir
                |--val_dir
                |--test_dir
        """
        if data_mode == "audio":
            train_root = osp.join(data_root, train_dir)
            self.train_dataset = base_dataset.ImageFolderDataset(train_root,
                                                                 transform=train_transform)
            if val_dir is not None:
                val_root = osp.join(data_root, val_dir)
                self.val_dataset = base_dataset.ImageFolderDataset(val_root,
                                                                   transform=val_transform)
            if test_dir is not None:
                test_root = osp.join(data_root, test_dir)
                self.test_dataset = base_dataset.ImageFolderDataset(test_root,
                                                                    transform=test_transform)
        elif data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")
        else:
            raise Exception(
                "Please specify in the json a specified mode in data_mode")
