from cv2 import BORDER_REFLECT
from albumentations.pytorch.transforms import ToTensorV2

from albumentations import Resize, ShiftScaleRotate, HorizontalFlip, \
    ElasticTransform, GridDistortion, CoarseDropout, CLAHE, RandomBrightnessContrast, \
    RandomGamma, IAASharpen, Blur, MotionBlur, OneOf, Compose


def get_augmentations(phase: str, image_shape: tuple) -> Compose:
    """
    Return Compose object with list of augmentations
    :param phase: 'train' or 'val'
    :param image_shape: tuple with image shape for input to neural network
    :return:
    """
    list_transforms = []

    list_transforms.extend(
        [
            Resize(image_shape[0], image_shape[1]),
            # Normalize(mean=(69.0932, 69.3587, 68.9373), std=(48.4471, 48.4580, 48.,4263)),
            ToTensorV2()
        ]
    )

    if phase == 'train':
        list_transforms.extend(
            [
                ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, always_apply=True,
                    border_mode=BORDER_REFLECT
                ),

                HorizontalFlip(),

                ElasticTransform(
                    p=1.0, alpha=1.4, sigma=13, alpha_affine=7, interpolation=0, border_mode=1, approximate=False
                ),

                GridDistortion(
                    p=0.8, num_steps=5, distort_limit=(-0.50, 0.5), interpolation=0, border_mode=1
                ),

                CoarseDropout(
                    p=1.0, max_holes=14, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8
                ),

                OneOf(
                    [
                        CLAHE(clip_limit=(1, 2.5)),
                        RandomBrightnessContrast(),
                        RandomGamma(),
                    ],
                    p=1,
                ),

                OneOf(
                    [
                        IAASharpen(p=1),
                        Blur(blur_limit=3, p=1),
                        MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.9,
                ),
            ]
        )

    list_transforms = Compose(list_transforms)
    return list_transforms
