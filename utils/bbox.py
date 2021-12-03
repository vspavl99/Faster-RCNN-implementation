import torch
from torch import Tensor

def get_target_shift(bboxes, anchors) -> list:
    """
    Calculate and return target shift for each proposals
    :param bboxes:
    :param anchors:
    :return:
    """

    return []


def get_proposals_from_bbox_regression(bbox_shifts: Tensor, bbox: list[Tensor]) -> Tensor:
    """
    Generate proposals from bbox and their shifts.
    :param bbox_shifts: [batch_size, w * h * anchors_number, 4]
    :param bbox: List[Tensor[self.anchors_number * w * h, 4], ... batch_size] (xyxy)
    :return: torch.Tensor
    """

    bbox = torch.stack(bbox, dim=0)

    assert bbox.shape == bbox_shifts.shape

    x_shift = bbox_shifts[:, :, 0]
    y_shift = bbox_shifts[:, :, 1]
    width_shift = bbox_shifts[:, :, 2]
    height_shift = bbox_shifts[:, :, 3]

    bbox_x = bbox[:, :, 0]
    bbox_y = bbox[:, :, 1]
    bbox_width = bbox[:, :, 2]
    bbox_height = bbox[:, :, 3]

    proposals = torch.empty_like(bbox_shifts)

    proposals[:, :, 0] = bbox_width * x_shift + bbox_x
    proposals[:, :, 1] = bbox_height * y_shift + bbox_y
    proposals[:, :, 2] = torch.exp(width_shift) * bbox_width
    proposals[:, :, 3] = torch.exp(height_shift) * bbox_height

    return proposals
