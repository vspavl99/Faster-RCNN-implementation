import torch
from torch import Tensor
from typing import List, Tuple
from torchvision.ops import box_iou


class BBoxMatcher:
    def __init__(self, low_threshold: float, high_threshold: float):

        self.LOW_MARKER = -1
        self.BETWEEN_MARKER = -2

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def match(self, proposals: Tensor, ground_true_bboxes: Tensor):
        """

        :param proposals:
        :param ground_true_bboxes:
        :return:
        """

        ious = box_iou(proposals, ground_true_bboxes)

