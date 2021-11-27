import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torch import Tensor


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.anchors_sizes = (32, 64)
        self.aspect_ratios = ((0.5, 1.0,),)

        self.anchors_number = len(self.anchors_sizes) * len(self.aspect_ratios)

        self.features_size = 512
        self.backbone = vgg16(pretrained=True).features

        self.top_n_proposal = 100

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

        self.anchor_generator = AnchorGenerator(sizes=(self.anchors_sizes, ), aspect_ratios=(self.aspect_ratios,))

    def filter_proposals(self, proposals, object_score):
        """
        Filtering porposals (clip)
        :param proposals:
        :param object_score:
        :return:
        """
        batch_size = object_score.shape[0]
        object_score = torch.reshape(object_score.detach().clone(), (batch_size, -1))

        if self.top_n_proposal is not None:
            top_n_proposal = min(self.top_n_proposal, proposals.shape[1])
            _, keep_proposal_indexes = torch.topk(object_score, top_n_proposal, dim=1)

            proposals = proposals[:, keep_proposal_indexes]
            objectnesses_prob = object_score[torch.arange(batch_size)[:, None], keep_proposal_indexes]


        return


    def forward(self, batch_images):
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

        return feature_map, object_score, bbox_regression


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
