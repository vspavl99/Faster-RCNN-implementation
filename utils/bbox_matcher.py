from torch import Tensor
from torchvision.ops import box_iou
from model.test import draw
import torch


class BBoxMatcher:
    def __init__(self, low_threshold: float = 0.5, high_threshold: float = 0.5):
        self.LOW_MARKER = -1
        self.BETWEEN_MARKER = -2

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def match(self, proposals: Tensor, ground_true_bboxes: Tensor):
        """

        :param proposals: (xyxy)
        :param ground_true_bboxes: must be in (xyxy)
        :return:
        """

        ious = box_iou(proposals, ground_true_bboxes)

        # Find biggest IoU with ground_true_bbox for every proposals
        iou_values, gt_bbox_indexes = ious.max(dim=1)

        # draw(proposals, ground_true_bboxes)

        gt_bbox_indexes[iou_values < self.low_threshold] = self.LOW_MARKER

        # cross_boundary = get_cross_boundary_box_idxs(proposals, (224, 224))
        # gt_bbox_indexes[cross_boundary] = self.BETWEEN_MARKER

        gt_bbox_indexes[
            (iou_values >= self.low_threshold) & (iou_values < self.high_threshold)
            ] = self.BETWEEN_MARKER

        return gt_bbox_indexes


def draw(proposals, ground_true_bboxes):
    import cv2
    import numpy as np
    dummy_image = np.array(torch.ones(800, 800, 3, dtype=torch.uint8))

    for i, anchor in enumerate(proposals):
        x1, y1, x2, y2 = int(anchor[0].item()), int(anchor[1].item()), int(anchor[2].item()), int(anchor[3].item())
        dummy_image = cv2.rectangle(img=dummy_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255))

    for i, anchor in enumerate(ground_true_bboxes):
        x1, y1, x2, y2 = int(anchor[0].item()), int(anchor[1].item()), int(anchor[2].item()), int(anchor[3].item())
        # print(x1, y1, x2, y2)

        dummy_image = cv2.rectangle(img=dummy_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0))
        # break

    import matplotlib.pyplot as plt
    plt.imshow(dummy_image)
    plt.show()


def get_cross_boundary_box_idxs(boxes, image_size):
    """
    :param boxes: [K, 4]
    :param image_size: [2,]
    """
    mask = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
    mask = mask | (boxes[:, 0] < 0)
    mask = mask | (boxes[:, 0] > image_size[1])

    mask = mask | (boxes[:, 1] < 0)
    mask = mask | (boxes[:, 1] > image_size[0])

    mask = mask | (boxes[:, 2] < 0)
    mask = mask | (boxes[:, 2] > image_size[1])

    mask = mask | (boxes[:, 3] < 0)
    mask = mask | (boxes[:, 3] > image_size[0])

    idxs = mask.nonzero().squeeze(-1)
    return idxs