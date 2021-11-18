import torch
from torch import nn
from torchvision.models import vgg16


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.anchors_number = 9

        self.backbone = vgg16(pretrained=True)

        self.base = nn.Sequential(
            *list(self.backbone.features._modules.values())[:-1]
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.object_score = nn.Conv2d(
            in_channels=512, out_channels=(2 * self.anchors_number), kernel_size=(1, 1)
        )

        self.bbox_regression = nn.Conv2d(
            in_channels=512, out_channels=(4 * self.anchors_number), kernel_size=(1, 1)
        )

    def forward(self, input):
        features = self.base(input)

        backbone_features = self.conv1(features)
        object_score, bbox_regression = self.object_score(backbone_features), self.bbox_regression(backbone_features)

        return features, object_score, bbox_regression


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 800, 800))
    print(dummy_input.shape)
    rpn = RPN()
    res = rpn(dummy_input)
    print(res[1].shape, res[2].shape)

# class FasterRCNN(nn.Module):
#     def __init__(self):
#         super(FasterRCNN, self).__init__()
#         self.
#
#     def forward(self, input):
