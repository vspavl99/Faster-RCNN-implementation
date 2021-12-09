import torch
from torch import nn
from torchvision.ops import RoIPool
from torch import Tensor
from utils.bbox_matcher import BBoxMatcher
from utils.sampler import Sampler
from utils.bbox import get_proposals_from_bbox_regression, get_target_shift
from torch.nn import functional as F
from torchvision.ops import clip_boxes_to_image, remove_small_boxes, batched_nms


class FastRCNN(nn.Module):
    def __init__(self, stage: str = 'train'):
        super().__init__()

        self.n_class = 21
        self.spatial_scale = 1 / 32
        self.roi_pooling_size = 7
        self.feature_map_size = 512
        self.score_thresh = 0.1
        self.nms_thresh = 0.3
        self.detections_per_img = 100

        self.roi_pooling = RoIPool(output_size=self.roi_pooling_size, spatial_scale=self.spatial_scale )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.feature_map_size * self.roi_pooling_size ** 2, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.fc_objects = nn.Linear(1024, self.n_class)
        self.fc_scores = nn.Linear(1024, self.n_class * 4)

        self.bbox_matcher = BBoxMatcher()
        self.sampler = Sampler()

        self.training = True if stage == 'train' else False

    def forward(self, feature_map: Tensor, proposals: list[Tensor], targets: list[dict]):
        labels = None
        regression_targets = None
        original_image_shape = feature_map.shape[1] // self.spatial_scale,  feature_map.shape[2] // self.spatial_scale

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
            losses = {"fastrcnn_loss_classifier": loss_classifier, "fastrcnn_loss_box_reg": loss_box_reg}

        else:
            boxes, scores, labels = self.postprocess_detections(
                features_class, features_scores, proposals, original_image_shape
            )
            for boxes_per_image, scores_per_image, labels_per_image in zip(boxes, scores, labels):
                result.append({"boxes": boxes_per_image, "labels": scores_per_image, "scores": labels_per_image})

        return result, losses

    @staticmethod
    def loss(features_class: Tensor, features_scores: Tensor, batch_labels: list[Tensor],
             regression_targets: list[Tensor]) -> tuple:
        """
        :param features_class: # TODO:
        :param features_scores:
        :param batch_labels:
        :param regression_targets:
        :return:
        """

        labels = torch.cat(batch_labels, dim=0)
        classification_loss = F.cross_entropy(features_class, labels)

        positive_indexes = torch.where(labels > 0)[0]
        labels_pos = labels[positive_indexes]

        regression_targets = torch.cat(regression_targets, dim=0)
        regression_targets = regression_targets.reshape(-1, 4)
        regression_targets_positive = regression_targets[positive_indexes]

        features_scores = features_scores.reshape(features_scores.shape[0], -1, 4)

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
            matched_gt_bboxes.append(gt_boxes_per_batch[matched_indexes_per_batch[sample_indexes_per_batch]])

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

    def postprocess_detections(self, features_class: Tensor, features_scores: Tensor,
                               batch_proposals: list[Tensor], image_shape: tuple):
        """
        :param features_class:
        :param features_scores:
        :param proposals:
        :param image_shape:
        :return:
        """
        device = features_class.device

        #TODO: don't work
        predicted_bboxes = get_proposals_from_bbox_regression(features_scores, batch_proposals)

        proposals_per_image = [proposals_per_image.shape[0] for proposals_per_image in batch_proposals]
        predicted_bboxes = predicted_bboxes.split(proposals_per_image, 0)

        predicted_scores = F.softmax(features_class, -1)
        predicted_scores = predicted_scores.split(proposals_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores in zip(predicted_bboxes, predicted_scores):
            boxes = clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(self.n_class, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


if __name__ == '__main__':
    dummy_input = torch.randn((2, 3, 200, 200))

    from torchvision.models import vgg16
    from model.rpn import RPN
    rpn = RPN(stage='val')
    feature_map, filtered_proposals, loss_object_score, loss_proposals = rpn(dummy_input)

    fast_rcnn = FastRCNN()
    targets = [
            {'boxes': torch.tensor([[50, 50, 150, 150]]), 'labels': torch.tensor([0])},
            {'boxes': torch.tensor([[50, 50, 150, 150]]), 'labels': torch.tensor([0])}
        ]
    output = fast_rcnn(feature_map, filtered_proposals, targets)

    print(dummy_input.shape)

    from model.rpn import RPN
    rpn = RPN()
    optimizer = torch.optim.Adam(rpn.parameters())

    for i in range(100):
        dummy_input = torch.ones((2, 3, 800, 800))

        feature_map, proposals, loss_object_score, loss_bbox = rpn(
            dummy_input, torch.tensor([[[50, 50, 150, 150]], [[50, 50, 150, 150]]])
        )
        loss = loss_object_score + loss_bbox
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
