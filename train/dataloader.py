from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from utils.augmentations import get_augmentations
import cv2
import torch

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigDataset:
    image_shape: tuple = (100, 100)
    image_directory: Path = Path('data/PASCAL VOC 2012/JPEGImages')
    path_to_csv: Path = Path('data/annotation.csv')

    classes = (
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    )


class DatasetFasterRCNN(Dataset):
    def __init__(self, config_dataset: ConfigDataset, phase='train'):

        self.phase: str = phase

        self.image_shape: tuple = config_dataset.image_shape
        self.image_directory: Path = config_dataset.image_directory
        self.path_to_csv_annotation: Path = config_dataset.path_to_csv
        self.classes = config_dataset.classes

        # Set augmentations
        self.transforms = get_augmentations(phase=self.phase, image_shape=self.image_shape)

        # Read annotations from csv
        self.dataset: pd.DataFrame = pd.read_csv(self.path_to_csv_annotation)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        # TODO: Measure time for this operation
        item = self.dataset.iloc[index]

        image = cv2.imread(self.image_directory.joinpath(item['file_name']))

        assert image is not None, item

        # Apply transforms
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']

        # TODO: Define target bbox and class
        class_name = item['class']
        target = torch.tensor(self.classes == class_name).float().argmax()
        return image, torch.tensor(target)

    def __len__(self):
        return len(self.dataset)


def get_dataloader(path_train_csv: Path, path_val_csv: Path, shuffle: bool, batch_size: int = 8) -> dict:
    """
    Return dict with train and val dataloader
    :param path_train_csv: Path to the train annotations
    :param path_val_csv: Path to the val annotations
    :param shuffle: shuffle data
    :param batch_size: batch_size
    :return:
    """
    dataloader = {
        'train': DataLoader(
            dataset=DatasetFasterRCNN(ConfigDataset(path_to_csv=path_train_csv), phase='train'),
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'val': DataLoader(
            dataset=DatasetFasterRCNN(ConfigDataset(path_to_csv=path_val_csv), phase='val'),
            batch_size=batch_size,
            shuffle=shuffle
        )
    }

    return dataloader
