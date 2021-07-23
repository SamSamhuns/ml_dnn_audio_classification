from torchvision import transforms


class Preprocess:
    common_transform = transforms.Compose([
        transforms.ToTensor()])
    train = common_transform
    val = common_transform
    test = common_transform
    inference = common_transform

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        pass
