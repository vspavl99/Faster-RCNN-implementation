import torch
from torch import nn
from torchvision.ops import RoIPool
from torch import Tensor
from utils.bbox_matcher import BBoxMatcher
from utils.sampler import Sampler


class FastRCNN(nn.Module):
    def __init__(self):
        super(FastRCNN, self).__init__()

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

    def forward(self, feature_map: Tensor, regions_of_interest: list[Tensor], targets: list[dict]):

        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(regions_of_interest, gt_boxes)
        ]

        batch_labels, batch_matched_indexes = self.assign_targets_to_anchors(proposals, gt_boxes, gt_labels)

        _, _, sampled_inds = self.sampler.create_minibatch(batch_labels)

        # feature_map (Tensor[N, C, H, W])
        features = self.roi_pooling(feature_map, regions_of_interest)

        features = self.fc(features)

        # features_class, features_scores = self.fc_objects(features),  self.fc_scores(features)
        features_class, features_scores = None, None
        return features_class, features_scores

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


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 200, 200))

    from torchvision.models import vgg16
    from model.rpn import RPN
    rpn = RPN(stage='val')
    feature_map, filtered_proposals, loss_object_score, loss_proposals = rpn(dummy_input)

    fast_rcnn = FastRCNN()
    targets = [
        {'boxes': torch.tensor([[50, 50, 150, 150]]), 'labels': torch.tensor([0])},
        # {'boxes': torch.tensor([[50, 50, 150, 150]]), 'labels': torch.tensor([0])}
        ]
    output = fast_rcnn(feature_map, filtered_proposals, targets)

    print(dummy_input.shape)

