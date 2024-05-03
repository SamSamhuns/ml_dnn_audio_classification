# Pytorch Project Template, Audio

This is a template for a PyTorch Project for training, testing, inference demo, and FastAPI serving along with Docker support.

- [Pytorch Project Template, Audio](#pytorch-project-template-audio)
  - [Setup](#setup)
  - [Train](#train)
  - [Custom Training](#custom-training)
    - [Audio Classification](#audio-classification)
  - [Test](#test)
  - [Tensorboard logging](#tensorboard-logging)
  - [Inference](#inference)
  - [Docker](#docker)
    - [For training and testing](#for-training-and-testing)
    - [Serving the model with FastAPI](#serving-the-model-with-fastapi)
  - [Utility functions](#utility-functions)
  - [Acknowledgements](#acknowledgements)


## Setup

Use `python venv` or a `conda env` to install requirements:

-   Install full-requirements: `pip install -r requirements.txt`
-   Install train/minimal requirements: `pip install -r requirements/train|minimal.txt`

## Train

Example training for mnist digit classification with default config:

```shell
$ python train.py
```

## Custom Training

### Audio Classification

A possible smaple train and test dataset is the [urbansound8k](https://urbansounddataset.weebly.com/urbansound8k.html) dataset. Set training data inside `data` directory in the following format:

    data
    |── CUSTOM_DATASET
        ├── CLASS 1
        |   ├── audio1
        |   └── audio2
        |   ├── ...
        ├── CLASS 2
        |   ├── audio1
        |   └── audio2
        |   ├── ...

```shell
# create train val test split
$ python modules/utils/train_val_test_split.py -rd data/CUSTOM_DATASET -td data/CUSTOM_DATASET_SPLIT -vs VAL_SPLIT_FRAC -ts TEST_SPLIT_FRAC
# OPTIONAL duplicate train data if necessary
$ python modules/utils/duplicate_data.py -rd data/CUSTOM_DATASET_SPLIT/train -td data/CUSTOM_DATASET_SPLIT/train -n TARGET_NUMBER
# create a custom config file based on configs/classifier_config.py and modify train parameters
$ cp configs/classifier_config.py configs/custom_classifier_config.py
# train on custom data with custom config
$ python train.py -c custom_classifier_config.py
```

## Test

Test based on CONFIG_FILE. By default testing is done for mnist classification.

```shell
$ python test.py -c CONFIG_FILE
```

## Tensorboard logging

All tensorboard logs are saved in the `TENSORBOARD_EXPERIMENT_DIR` setting in the config file. Logs include train/val epoch accuracy/loss, graph, and preprocessed images per epoch.

To start a tensorboard server reading logs from the `experiment` dir exposed on port localhost port `6007`:

```shell
$ tensorboard --logdir=experiments --port=6006
```

## Inference

## Docker

Install docker in the system first:

### For training and testing

```shell
$ bash build_docker.sh  # builds the docker image
$ bash run_docker.sh    # runs the previous docker image creating a shared volume checkpoint_docker outside the container
# inside the docker container
$ python train.py
```

Using gpus inside docker for training/testing:

`--gpus device=0,1 or all`

### Serving the model with FastAPI

```shell
$ bash server/build_docker.sh
$ bash server/run_docker.sh -h/--http 8080
```

## Utility functions

To cleanup:

    bash modules/utils/cleanup.sh

To copy project structure:

    $ python3 new_project.py ../NewProject

## Acknowledgements

-   <https://github.com/victoresque/pytorch-template>
