import torch
from torch import Tensor
from torchvision.ops import box_convert
from typing import List


def get_target_shift(bboxes, anchors) -> List[Tensor]:
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

    targets = []
    for boxes_per_image, anchor_per_image in zip(bboxes, anchors):

        boxes_per_image = box_convert(boxes_per_image, 'xyxy', out_fmt='cxcywh')
        anchor_per_image = box_convert(anchor_per_image, 'xyxy', out_fmt='cxcywh')

        targets_per_image = torch.empty_like(anchor_per_image)
        targets_per_image[:, 0] = (boxes_per_image[:, 0] - anchor_per_image[:, 0]) / anchor_per_image[:, 2]
        targets_per_image[:, 1] = (boxes_per_image[:, 1] - anchor_per_image[:, 1]) / anchor_per_image[:, 3]
        targets_per_image[:, 2] = torch.log(boxes_per_image[:, 2] / anchor_per_image[:, 2])
        targets_per_image[:, 3] = torch.log(boxes_per_image[:, 3] / anchor_per_image[:, 3])

        assert torch.isinf(targets_per_image).sum() == 0

        targets.append(targets_per_image)

    return targets


def get_proposals_from_bbox_regression(bbox_shifts: Tensor, bbox: List[Tensor]) -> Tensor:
    """
    Generate proposals from bbox and their shifts.
    :param bbox_shifts: [batch_size, w * h * anchors_number, 4] (XcYcWH)
    :param bbox: List[Tensor[self.anchors_number * w * h, 4], ... batch_size] (XYXY)
    :return: torch.Tensor
    """

    batch_size = None
    if len(bbox_shifts.shape) == 3:
        batch_size = bbox_shifts.shape[0]

    bbox_shifts = bbox_shifts.reshape(-1, bbox_shifts.shape[-1])
    bbox = torch.cat(bbox, dim=0)

    x_shift = bbox_shifts[:, 0::4]
    y_shift = bbox_shifts[:, 1::4]
    width_shift = bbox_shifts[:, 2::4]
    height_shift = bbox_shifts[:, 3::4]

    bbox_width = bbox[:, 2] - bbox[:, 0]
    bbox_height = bbox[:, 3] - bbox[:, 1]
    bbox_center_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_center_y = bbox[:, 1] + 0.5 * bbox_height

    proposals_x = bbox_width[:, None] * x_shift + bbox_center_x[:, None]
    proposals_y = bbox_height[:, None] * y_shift + bbox_center_y[:, None]
    proposals_w = torch.exp(width_shift) * bbox_width[:, None]
    proposals_h = torch.exp(height_shift) * bbox_height[:, None]

    proposals = torch.stack((proposals_x, proposals_y, proposals_w, proposals_h), dim=2).flatten(1)

    if proposals.shape[-1] == 4:
        proposals = box_convert(proposals, 'cxcywh', out_fmt='xyxy')
    else:
        proposals = box_convert(proposals.reshape(proposals.shape[0], -1, 4), 'cxcywh', out_fmt='xyxy')
        # proposals = proposals.reshape(proposals.shape[0], -1)

    if batch_size is not None:
        proposals = proposals.reshape(batch_size, -1, 4)

    return proposals
