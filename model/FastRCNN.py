import torch
from torch import nn
from torchvision.ops import RoIPool
from torch import Tensor
from utils.bbox_matcher import BBoxMatcher
from utils.sampler import Sampler
from utils.bbox import get_proposals_from_bbox_regression, get_target_shift
from torch.nn import functional as F


class FastRCNN(nn.Module):
    def __init__(self, stage: str = 'train'):
        super().__init__()

        self.n_class = 20

        self.roi_pooling = RoIPool(output_size=(3, 3), spatial_scale=1 / 32)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=9 * 512, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc_objects = nn.Linear(1024, self.n_class)
        self.fc_scores = nn.Linear(1024, self.n_class * 4)

        self.bbox_matcher = BBoxMatcher()
        self.sampler = Sampler()

        self.training_stage = True if stage == 'train' else False

    def forward(self, feature_map: Tensor, proposals: list[Tensor], targets: list[dict]):
        labels = None
        regression_targets = None

        if self.training:
            gt_boxes = [t["boxes"] for t in targets]
            gt_labels = [t["labels"] for t in targets]

            # append ground-truth bboxes to proposals
            proposals = [
                torch.cat((proposal, gt_box))
                for proposal, gt_box in zip(proposals, gt_boxes)
            ]

            batch_labels, batch_matched_indexes = self.assign_targets_to_anchors(proposals, gt_boxes, gt_labels)

            labels, proposals, matched_gt_bboxes, matched_indexes = self.choose_samples(
                batch_labels, proposals, gt_boxes, batch_matched_indexes)

            regression_targets = get_target_shift(matched_gt_bboxes, proposals)

        # feature_map (Tensor[N, C, H, W])
        features = self.roi_pooling(feature_map, proposals)
        features = self.fc(features)
        features_class, features_scores = self.fc_objects(features),  self.fc_scores(features)

        result: list[dict[str, Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = self.loss(features_class, features_scores, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        else:

            boxes, scores, labels = self.postprocess_detections(features_class, features_scores, proposals, image_shapes)
            for boxes_per_image, scores_per_image, labels_per_image in zip(boxes, scores, labels):
                result.append({"boxes": boxes_per_image, "labels": scores_per_image, "scores": labels_per_image})

        return result, losses

    def loss(self, features_class: Tensor, features_scores: Tensor, batch_labels: list[Tensor],
             regression_targets: Tensor) -> tuple:
        """

        :param features_class:
        :param features_scores:
        :param batch_labels:
        :param regression_targets:
        :return:
        """

        labels = torch.cat(batch_labels, dim=0)
        regression_targets = regression_targets.reshape(-1, 4)

        classification_loss = F.cross_entropy(features_class, labels)

        positive_indexes = torch.where(labels > 0)[0]
        labels_pos = labels[positive_indexes]
        features_scores = features_scores.reshape(features_scores.shape[0], -1, 4)

        regression_targets_positive = regression_targets[positive_indexes]
        # features_scores

        regression_loss = F.smooth_l1_loss(
            features_scores[positive_indexes, labels_pos],
            regression_targets_positive,
            reduction='mean'
        )

        return classification_loss, regression_loss

    def choose_samples(self, batch_labels: list[Tensor], batch_proposals: list[Tensor],
                       batch_gt_boxes: list[Tensor], batch_matched_indexes: list[Tensor]) -> tuple:
        """
        Select minibatch samples
        :param batch_matched_indexes:
        :param batch_labels:
        :param batch_proposals:
        :param batch_gt_boxes:
        :return:
        """

        dtype = batch_proposals[0].dtype
        device = batch_proposals[0].device

        _, _, batch_sampled_indexes = self.sampler.create_minibatch(batch_labels)

        labels, proposals, matched_gt_bboxes, matched_indexes = [], [], [], []

        for sample_indexes_per_batch, labels_per_batch, gt_boxes_per_batch,\
            proposals_per_batch, matched_indexes_per_batch in \
                zip(batch_sampled_indexes, batch_labels, batch_gt_boxes, batch_proposals, batch_matched_indexes):

            labels.append(labels_per_batch[sample_indexes_per_batch])
            proposals.append(proposals_per_batch[sample_indexes_per_batch])
            matched_indexes.append(matched_indexes_per_batch[sample_indexes_per_batch])

            if gt_boxes_per_batch.numel() == 0:
                gt_boxes_per_batch = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_bboxes.append(gt_boxes_per_batch[matched_indexes_per_batch])

        return labels, proposals, matched_gt_bboxes, matched_indexes

    def assign_targets_to_anchors(
            self, batch_proposals: list[Tensor], batch_ground_true_bboxes: list[Tensor],
            batch_ground_true_labels: list[Tensor]) -> tuple[list, list]:
        """
        For every proposals find ground_true_bboxes with highest IoU. And assigns labels them
        :param batch_proposals: batch of anchors
        :param batch_ground_true_bboxes: batch of ground_true_bboxes
        :param batch_ground_true_labels:
        :return: batch_ground_true_labels, ground_true_bbox_per_proposal
        """
        assert len(batch_proposals) == len(batch_ground_true_bboxes)

        batch_matched_indexes = []
        batch_labels = []

        for proposals, ground_true_bboxes, ground_true_labels in zip(
                batch_proposals, batch_ground_true_bboxes, batch_ground_true_labels
        ):
            matched_indexes = self.bbox_matcher.match(proposals, ground_true_bboxes)

            if ground_true_bboxes.numel() == 0:
                device = proposals.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals.shape[0],), dtype=torch.int64, device=device
                )
                labels = torch.zeros(
                    (proposals.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # Set 0 ground_true index for proposals with low IoU
                clamped_matched_idxs_in_image = matched_indexes.clamp(min=0)
                labels = ground_true_labels[clamped_matched_idxs_in_image].to(dtype=torch.int64)

                # Negative examples
                labels[matched_indexes == self.bbox_matcher.LOW_MARKER] = 0

                # Between proposals
                labels[matched_indexes == self.bbox_matcher.BETWEEN_MARKER] = -1

            batch_labels.append(labels)
            batch_matched_indexes.append(clamped_matched_idxs_in_image)

        return batch_labels, batch_matched_indexes
