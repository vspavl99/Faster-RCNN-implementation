import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import remove_small_boxes, clip_boxes_to_image, nms
from torch.nn import SmoothL1Loss, BCEWithLogitsLoss
from torch.nn import functional as F

from torch import Tensor
from utils.bbox_matcher import BBoxMatcher
from utils.sampler import Sampler
from utils.bbox import get_proposals_from_bbox_regression, get_target_shift


class RPN(nn.Module):
    def __init__(self, stage: str = 'train'):
        super(RPN, self).__init__()

        # Anchors parameters
        self.anchors_sizes = (32,)
        self.aspect_ratios = ((1.0,),)
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
        self.post_nms_top_n_proposal = 500
        self.pre_nms_top_n_proposal = 500
        self.nms_threshold = 0.7
        self.image_size = (800, 800)

        # Training
        self.training_stage = True if stage == 'train' else False
        self.bbox_matcher = BBoxMatcher(0.1, 0.01)
        self.sampler = Sampler()

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

        proposals = clip_boxes_to_image(proposals, self.image_size)

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

    def assign_targets_to_anchors(
            self, batch_anchors: list[Tensor], batch_ground_true_bboxes: list[Tensor]) -> tuple[list, list]:
        """
        For every proposals find ground_true_bboxes with highest IoU. And assigns labels them
        :param batch_anchors: batch of anchors
        :param batch_ground_true_bboxes: batch of ground_true_bboxes
        :return: labels, ground_true_bbox_per_proposal
        """
        assert len(batch_anchors) == len(batch_ground_true_bboxes)

        batch_matched_gt_bboxes_to_anchors = []
        batch_labels = []

        for anchors, ground_true_bboxes in zip(batch_anchors, batch_ground_true_bboxes):
            matched_indexes = self.bbox_matcher.match(anchors, ground_true_bboxes)

            # Set 0 ground_true index for proposals with low IoU
            matched_gt_bboxes_to_anchors = ground_true_bboxes[matched_indexes.clamp(min=0)]

            labels = (matched_indexes >= 0).to(dtype=torch.float32)

            # Negative examples
            labels[matched_indexes == self.bbox_matcher.LOW_MARKER] = 0

            # Between proposals
            labels[matched_indexes == self.bbox_matcher.BETWEEN_MARKER] = -1

            batch_labels.append(labels)
            batch_matched_gt_bboxes_to_anchors.append(matched_gt_bboxes_to_anchors)

        return batch_labels, batch_matched_gt_bboxes_to_anchors

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
            assert batch_targets is not None

            labels, matched_gt_bboxes = self.assign_targets_to_anchors(anchors, batch_targets)
            target_shift = get_target_shift(matched_gt_bboxes, anchors)
            loss_object_score, loss_proposals = self.loss(
                target_shift, labels, object_score, bbox_regression,
            )

        else:
            loss_object_score, loss_proposals = None, None

        return feature_map, filtered_proposals, loss_object_score, loss_proposals

    def loss(self, target: Tensor, labels: list[Tensor], object_score: Tensor, bbox_regression: Tensor) -> tuple:
        """
        Take mini-batch of anchors and compute loss for them
        :param target:
        :param labels:
        :param object_score:
        :param bbox_regression:
        :return:
        """
        batch_size = len(object_score)
        positive_samples, negative_samples, samples = self.sampler.create_minibatch(labels)

        # TODO: choose implementation
        # flatten indexes per all batch in flatten
        # sampled_pos_inds = torch.where(torch.cat(positive_samples, dim=0))[0]
        # sampled_neg_inds = torch.where(torch.cat(negative_samples, dim=0))[0]
        # sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0) # flatten indexes
        # objectness = object_score.flatten()
        # tmp3 = objectness[sampled_inds]

        #  samples = indexes per every batch
        object_score_minibatch = [
            object_score.reshape(batch_size, -1)[batch_number, indexes] for batch_number, indexes in enumerate(samples)
        ]
        labels_minibatch = [
            labels[batch_number][indexes] for batch_number, indexes in enumerate(samples)
        ]

        bbox_regression_minibatch = [
            bbox_regression[batch_number, indexes, :] for batch_number, indexes in enumerate(samples)
        ]
        target_regression_minibatch = [
            target[batch_number, indexes, :] for batch_number, indexes in enumerate(samples)
        ]

        # Convert all list of Tensors into Tensors
        object_score_minibatch = torch.cat(object_score_minibatch, dim=0)
        labels_minibatch = torch.cat(labels_minibatch, dim=0)

        target_regression_minibatch = torch.cat(target_regression_minibatch, dim=0)
        bbox_regression_minibatch = torch.cat(bbox_regression_minibatch, dim=0)

        box_loss = F.smooth_l1_loss(
            bbox_regression_minibatch, target_regression_minibatch, reduction='sum'
        )

        objectness_loss = F.binary_cross_entropy_with_logits(
            object_score_minibatch, labels_minibatch, reduction='mean'
        )

        return box_loss, objectness_loss


if __name__ == '__main__':
    dummy_input = torch.randn((2, 3, 664, 664))

    rpn = RPN()
    res = rpn(dummy_input, torch.tensor([[[50, 50, 150, 150]], [[50, 50, 150, 150]]]))
    print(res[0].shape, len(res[1]), res[1][0].shape, res[2], res[3])
