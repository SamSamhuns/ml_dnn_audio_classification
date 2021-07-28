# imported objects should not be instanitated here
from torch import nn
import torch.optim as optim

from modules.models.audio_classifier_model import AudioClassifier
from modules.augmentations.audio_transforms import Preprocess
from modules.dataloaders.base_dataloader import BaseDataLoader
from modules.datasets.urban_sound_8k_dataset import UrbanSound8KDataset


CONFIG = {
    "NAME": "audio_classifier",
    "SEED": 1,
    "USE_CUDA": False,            # set to True for gpu training
    "CUDNN_DETERMINISTIC": True,  # for repeating results together with SEED
    "CUDNN_BENCHMARK": False,     # set to True for faster training with gpu
    "GPU_DEVICE": [0],            # cuda device list for single/multi gpu training
    "USE_AMP": False,             # automatic mixed precision training for faster train
    "ARCH": {
        "TYPE": AudioClassifier,
        "ARGS": {},
        "INPUT_WIDTH": 344,
        "INPUT_HEIGHT": 64,
        "INPUT_CHANNEL": 2
    },
    "DATASET": {
        "TYPE": UrbanSound8KDataset,
        "NUM_CLASSES": 10,
        "DATA_DIR": {"data_root": "data/UrbanSound8K/audio",
                     "csv_annot_root": "data/UrbanSound8K/metadata/UrbanSound8K.csv",
                     "train_dir": "placeholder",
                     "val_dir": "placeholder",
                     "test_dir": "placeholder",
                     },
        "PREPROCESS": {"transform": Preprocess.train,
                       "target_transform": Preprocess.target,
                       "val_split": 0.1,
                       "test_split": 0.1},
    },
    "DATALOADER": {
        "TYPE": BaseDataLoader,
        "ARGS": {"batch_size": 32,
                 "shuffle": True,
                 "num_workers": 0,
                 "validation_split": 0.,  # set to 0 for urban_sound_8k_dataset
                 "pin_memory": True,
                 "drop_last": False,
                 "prefetch_factor": 2,
                 "worker_init_fn": None
                 },
    },
    "OPTIMIZER": {
        "TYPE": optim.Adam,
        "ARGS": {"lr": 1e-3}
    },
    "LOSS": nn.CrossEntropyLoss,
    "METRICS": ["val_accuracy"],
    "LR_SCHEDULER": {
        "TYPE": optim.lr_scheduler.ReduceLROnPlateau,
        "ARGS": {"factor": 0.1,
                 "patience": 8}
    },
    "TRAINER": {
        "RESUME": True,
        "SAVE_BEST_ONLY": True,
        "LOG_FREQ": 5,
        "VALID_FREQ": 2,

        "EPOCHS": 12,
        "CHECKPOINT_DIR": "checkpoints_urban_sound_8k",

        "VERBOSITY": 2,
        "EARLY_STOP": 10,
        "USE_TENSORBOARD": True,
        "TENSORBOARD_EXPERIMENT_DIR": "experiments_urban_sound_8k",
        "TENSORBOARD_PORT": 6006
    },
    "LOGGER": {
        "DIR": "logs_urban_sound_8k",
        "LOG_FMT": "classifier_log_{}.txt",
        "FILE_FMT": "%(asctime)s %(levelname)-8s: %(message)s",
        "CONSOLE_FMT": "%(message)s",

        "<logger levels>": "DEBUG:10, INFO:20, ERROR:40",
        "LOGGER_LEVEL": 10,
        "FILE_LEVEL": 10,
        "CONSOLE_LEVEL": 10
    }
}
