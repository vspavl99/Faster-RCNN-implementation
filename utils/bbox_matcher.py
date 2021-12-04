from torch import Tensor
from torchvision.ops import box_iou
from model.test import draw


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

        # TODO: Debug draw remove
        # draw(proposals, ground_true_bboxes)
        gt_bbox_indexes[iou_values < self.low_threshold] = self.LOW_MARKER

        gt_bbox_indexes[
            (iou_values >= self.low_threshold) & (iou_values < self.high_threshold)
            ] = self.BETWEEN_MARKER

        return gt_bbox_indexes
