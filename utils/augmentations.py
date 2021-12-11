from cv2 import BORDER_REFLECT
from albumentations.pytorch.transforms import ToTensorV2

from albumentations import Resize, ShiftScaleRotate, HorizontalFlip, \
    ElasticTransform, GridDistortion, CoarseDropout, CLAHE, RandomBrightnessContrast, \
    RandomGamma, IAASharpen, Blur, MotionBlur, OneOf, Compose, BboxParams,Normalize


def get_augmentations(phase: str, image_shape: tuple) -> Compose:
    """
    Return Compose object with list of augmentations
    :param phase: 'train' or 'val'
    :param image_shape: tuple with image shape for input to neural network
    :return:
    """
    list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                # HorizontalFlip(),
                # RandomBrightnessContrast(),
            ]
        )

    list_transforms.extend(
        [
            Resize(image_shape[0], image_shape[1]),
            # Normalize(),
            ToTensorV2()
        ]
    )

    list_transforms = Compose(
        list_transforms,
        bbox_params=BboxParams(format='pascal_voc', label_fields=['labels', 'area', 'iscrowd'])
    )

    return list_transforms
