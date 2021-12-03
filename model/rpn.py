import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import remove_small_boxes, clip_boxes_to_image, nms
from torch import Tensor
from utils.bbox_matcher import BBoxMatcher


class RPN(nn.Module):
    def __init__(self, stage: str = 'train'):
        super(RPN, self).__init__()

        # Anchors parameters
        self.anchors_sizes = (32, 64)
        self.aspect_ratios = ((0.5, 1.0,),)
        self.anchors_number = len(self.anchors_sizes) * len(self.aspect_ratios)
        self.anchor_generator = AnchorGenerator(sizes=(self.anchors_sizes,), aspect_ratios=(self.aspect_ratios,))

        # Backbone definition
        self.features_size = 512
        self.backbone = vgg16(pretrained=True).features

        # Layers definition
        self.common_feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.features_size, out_channels=self.features_size,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.ReLU()
        )

        self.object_score = nn.Conv2d(
            in_channels=self.features_size, out_channels=(1 * self.anchors_number), kernel_size=(1, 1)
        )

        self.bbox_regression = nn.Conv2d(
            in_channels=self.features_size, out_channels=(4 * self.anchors_number), kernel_size=(1, 1)
        )

        # FilterProposals
        self.min_boxes_size = 1e-3
        self.min_box_score = 0.0
        self.post_nms_top_n_proposal = 2000
        self.pre_nms_top_n_proposal = 2000
        self.nms_threshold = 0.7
        self.image_size = (800, 800)

        # Training
        self.training_stage = True if stage == 'train' else False
        self.bbox_matcher = BBoxMatcher()

    def filter_proposals(self, proposals, object_score):
        """
        Filtering porposals (clip)
        :param proposals:
        :param object_score:
        :return:
        """
        batch_size = object_score.shape[0]
        object_score = torch.reshape(object_score.detach(), (batch_size, -1))

        if self.pre_nms_top_n_proposal is not None:
            top_n_proposal = min(self.pre_nms_top_n_proposal, proposals.shape[1])
            _, keep_proposal_indexes = torch.topk(object_score, top_n_proposal, dim=1)

            # TODO: change torch.arange(batch_size)[:, None]
            proposals = proposals[torch.arange(batch_size)[:, None], keep_proposal_indexes]
            object_score = object_score[torch.arange(batch_size)[:, None], keep_proposal_indexes]

        object_prob = torch.sigmoid(object_score)

        # proposals = clip_boxes_to_image(proposals, self.image_size)

        final_proposals, final_score = [], []
        for batch_proposals, batch_scores in zip(proposals, object_prob):

            # Remove small boxes
            keep_indexes = remove_small_boxes(batch_proposals, self.min_boxes_size)
            batch_proposals, batch_scores = batch_proposals[keep_indexes], batch_scores[keep_indexes]

            # Remove boxes with low score
            keep_indexes = torch.where(batch_scores >= self.min_box_score)[0]
            batch_proposals, batch_scores = batch_proposals[keep_indexes], batch_scores[keep_indexes]

            # Non-maximum suppression
            keep_indexes = nms(batch_proposals, batch_scores, self.nms_threshold)

            # Keep only top predictions
            keep_indexes = keep_indexes[:self.post_nms_top_n_proposal]
            batch_proposals, batch_scores = batch_proposals[keep_indexes], batch_scores[keep_indexes]

            final_proposals.append(batch_proposals)
            final_score.append(batch_scores)

        return final_proposals, final_score

    def forward(self, batch_images, batch_targets=None):
        # Feed input images into backbone network
        feature_map = self.backbone(batch_images)

        # Feed feature_map into Conv2d and ReLU
        feature_map = self.common_feature_extractor(feature_map)

        # Feed feature_map into two branches for bbox_regression and object_score
        # [Batch_size, self.anchors_number * k, w, h]
        object_score, bbox_regression = self.object_score(feature_map), self.bbox_regression(feature_map)

        # Create anchors List[Tensor[self.anchors_number * w * h, 4], ... batch_size] (xyxy)
        anchors = self.anchor_generator(
            ImageList(batch_images, [image.shape for image in batch_images]),
            [feature_map]
        )

        # Convert bbox_regression from [batch_size, self.anchors_number * 4, w, h] to [batch_size, -1, 4]
        bbox_regression = torch.reshape(bbox_regression, (bbox_regression.shape[0], 4, -1)).permute((0, 2, 1))
        proposals = get_proposals_from_bbox_regression(bbox_regression, anchors)

        filtered_proposals, filtered_object_score = self.filter_proposals(proposals, object_score)

        if self.training_stage:
            self.assign_targets_to_anchors(anchors, batch_targets)

        return feature_map, object_score, bbox_regression

    def assign_targets_to_anchors(self, batch_anchors: list[Tensor], batch_ground_true_bboxes: list[Tensor]):

        assert len(batch_anchors) == len(batch_ground_true_bboxes)
        # batch_size = len(batch_ground_true_bboxes)

        matched_gt_bboxes_to_anchors = []
        for anchors, ground_true_bboxes in zip(batch_anchors, batch_ground_true_bboxes):
            matched_indexes = self.bbox_matcher(anchors, ground_true_bboxes)

            matched_gt_bboxes_to_anchors.append()


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


if __name__ == '__main__':
    dummy_input = torch.randn((2, 3, 800, 800))

    rpn = RPN()
    res = rpn(dummy_input)
    print(res[1].shape, res[2].shape)
