import torch
from torch import nn
from model.FastRCNN import FastRCNN
from model.rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, stage: str = 'train'):
        super(FasterRCNN, self).__init__()

        self.rpn = RPN(stage=stage)
        self.fast_rcnn = FastRCNN(stage=stage)

        self.training = True if stage == 'train' else False

    def forward(self, batch_input, batch_targets=None):
        batch_gt_boxes = None
        losses = {}

        if self.training:
            assert batch_targets is not None
            batch_gt_boxes = [t["boxes"] for t in batch_targets]

        features, roi, losses_rpn = self.rpn(batch_input, batch_gt_boxes)
        result, losses_fastrcnn = self.fast_rcnn(features, roi, batch_targets)
        losses.update(losses_rpn)
        losses.update(losses_fastrcnn)

        return result, losses


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 800, 800))
    print(dummy_input.shape)
    faster_rcnn = FasterRCNN(stage='val')
    res, loss = faster_rcnn(dummy_input)
    print(res, loss)
