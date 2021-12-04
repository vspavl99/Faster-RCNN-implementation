import torch
from torch import Tensor
from torchvision.ops import box_convert


def get_target_shift(bboxes, anchors) -> Tensor:
    """
    Calculate and return target shift for each proposals
    :param bboxes: true boxes for every anchors
    :param anchors: anchors
    :return: Tensor (xcxywh)
    """

    # anchor_widths = anchors[:, 2] - anchors[:, 0]
    # anchor_heights = anchors[:, 3] - anchors[:, 1]
    # anchor_center_x = anchors[:, 0] + 0.5 * anchor_widths
    # anchor_center_y = anchors[:, 1] + 0.5 * anchor_heights
    #
    # gt_widths = bboxes[:, 2] - bboxes[:, 0]
    # gt_heights = bboxes[:, 3] - bboxes[:, 1]
    # gt_center_x = bboxes[:, 0] + 0.5 * gt_widths
    # gt_center_y = bboxes[:, 1] + 0.5 * gt_heights
    #
    # targets_dx = (gt_center_x - anchor_center_x) / anchor_widths
    # targets_dy = (gt_center_y - anchor_center_y) / anchor_heights
    # targets_dw = torch.log(gt_widths / anchor_widths)
    # targets_dh = torch.log(gt_heights / anchor_heights)

    bboxes = torch.stack(bboxes, dim=0)
    anchors = torch.stack(anchors, dim=0)

    bboxes = box_convert(bboxes, 'xyxy', out_fmt='cxcywh')
    anchors = box_convert(anchors, 'xyxy', out_fmt='cxcywh')

    targets = torch.empty_like(anchors)
    targets[:, :, 0] = (bboxes[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2]
    targets[:, :, 1] = (bboxes[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]
    targets[:, :, 2] = torch.log(bboxes[:, :, 2] / anchors[:, :, 2])
    targets[:, :, 3] = torch.log(bboxes[:, :, 3] / anchors[:, :, 3])

    return targets


def get_proposals_from_bbox_regression(bbox_shifts: Tensor, bbox: list[Tensor]) -> Tensor:
    """
    Generate proposals from bbox and their shifts.
    :param bbox_shifts: [batch_size, w * h * anchors_number, 4] (XcYcWH)
    :param bbox: List[Tensor[self.anchors_number * w * h, 4], ... batch_size] (XYXY)
    :return: torch.Tensor
    """

    bbox = torch.stack(bbox, dim=0)

    assert bbox.shape == bbox_shifts.shape

    x_shift = bbox_shifts[:, :, 0]
    y_shift = bbox_shifts[:, :, 1]
    width_shift = bbox_shifts[:, :, 2]
    height_shift = bbox_shifts[:, :, 3]

    bbox_width = bbox[:, :, 2] - bbox[:, :, 0]
    bbox_height = bbox[:, :, 3] - bbox[:, :, 1]
    bbox_center_x = bbox[:, :, 0] + 0.5 * bbox_width
    bbox_center_y = bbox[:, :, 1] + 0.5 * bbox_height

    proposals = torch.empty_like(bbox_shifts)

    proposals[:, :, 0] = bbox_width * x_shift + bbox_center_x
    proposals[:, :, 1] = bbox_height * y_shift + bbox_center_y
    proposals[:, :, 2] = torch.exp(width_shift) * bbox_width
    proposals[:, :, 3] = torch.exp(height_shift) * bbox_height

    proposals = box_convert(proposals, 'cxcywh', out_fmt='xyxy')

    return proposals
