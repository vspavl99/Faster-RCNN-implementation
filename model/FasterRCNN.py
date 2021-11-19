import features as features
import torch
from torch import nn
from torchvision.ops import RoIPool
from model.FastRCNN import FastRCNN
from model.rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()

        self.rpn = RPN()
        self.fast_rcnn = FastRCNN()

    def forward(self, input):
        features, roi = self.rpn(input)

        out_class, out_bbox = self.fast_rcnn(features, roi)

        return out_class, out_bbox


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
