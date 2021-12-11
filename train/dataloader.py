import time

from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from utils.augmentations import get_augmentations
import cv2
import torch

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class ConfigDataset:
    image_shape: tuple = (800, 800)
    image_directory: Path = Path('../data/PASCAL VOC 2012/JPEGImages')
    path_to_csv: Path = Path('../data/annotation.csv')

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

        self.images_list = self.dataset['file_name'].unique()
        # self.annotations_per_image = self.get_annotations()

    def get_annotations(self):
        annotations_per_image = {}
        for image in self.images_list:
            objects_on_image = self.dataset[self.dataset['file_name'] == image]
            annotations = objects_on_image[['class', 'x1', 'y1', 'x2', 'y2']]
            annotations_per_image[image] = annotations

        return annotations_per_image

    def get_target(self, image_name):
        objects_on_image = self.dataset[self.dataset['file_name'] == image_name]
        annotations = objects_on_image[['class', 'x1', 'y1', 'x2', 'y2']]
        num_objects = len(annotations)

        bboxes = torch.as_tensor(
            [
                [obj['x1'], obj['y1'], obj['x2'], obj['y2']] for _, obj in annotations.iterrows()
            ], dtype=torch.float32
        )

        class_ids = torch.as_tensor(
            [self.classes.index(obj['class']) for _, obj in annotations.iterrows()]
        )

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        is_crowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {
            'bboxes': bboxes,
            'labels': class_ids,
            'area': area,
            'iscrowd': is_crowd
        }

        return target

    def __getitem__(self, index):
        image_name = self.images_list[index]

        image = cv2.imread(str(self.image_directory.joinpath(image_name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.get_target(image_name)

        # self.debug_augmentation(image, target)

        augmented_data = self.transforms(image=image, **target)

        image = augmented_data.pop('image')

        # self.debug_augmentation(image, augmented_data)

        augmented_data["image_id"] = torch.tensor([index])
        # augmented_data['boxes'] = torch.stack([torch.Tensor(bboxes) for bboxes in augmented_data['bboxes']])
        augmented_data['boxes'] = torch.as_tensor(augmented_data['bboxes'])
        augmented_data['labels'] = torch.as_tensor(augmented_data['labels'])
        augmented_data['area'] = torch.as_tensor(augmented_data['area'])
        augmented_data['iscrowd'] = torch.as_tensor(augmented_data['iscrowd'])


        return image, augmented_data

    def debug_augmentation(self, image, target):

        if isinstance(image, torch.Tensor):
            image = np.array(image, dtype=np.uint8)
            image = np.transpose(image, (1, 2, 0))

        for box, label, _, _ in zip(*target.values()):

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            image = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=2)
            image = cv2.putText(image, f"{self.classes[int(label)]}", (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        plt.imshow(image)
        plt.show()

    def __len__(self):
        return len(self.images_list)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(path_train_csv: Path, path_val_csv: Path, shuffle: bool, batch_size: int = 8):
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
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=4
        ),
        'val': DataLoader(
            dataset=DatasetFasterRCNN(ConfigDataset(path_to_csv=path_val_csv), phase='val'),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=4
        )
    }

    return dataloader


if __name__ == '__main__':
    pass
    dataset = DatasetFasterRCNN(ConfigDataset())
    dataloaders = get_dataloader(
        path_train_csv=Path('../data/annotation.csv'),
        path_val_csv=Path('../data/annotation2.csv'),
        shuffle=False, batch_size=20
    )

    i = 0
    a = time.time()
    for batch_images, batch_targets in dataloaders['train']:
        print(time.time() - a)
        a = time.time()
        i += 1
        if i == 100:
            break


