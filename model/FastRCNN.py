import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool


class FastRCNN(nn.Module):
    def __init__(self):
        super(FastRCNN, self).__init__()

        self.n_class = 20

        self.roi_pooling = RoIPool(output_size=(7, 7), spatial_scale=1 / 16)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc_objects = nn.Linear(4096, self.n_class)
        self.fc_scores = nn.Linear(4096, self.n_class * 4)

    def forward(self, feature_map, regions_of_interest):

        # feature_map (Tensor[N, C, H, W])
        features = self.roi_pooling(feature_map, regions_of_interest)

        features = self.fc1(features)
        features = self.fc2(features)

        features_class, features_scores = self.fc_objects(features),  self.fc_scores(features)

        return features_class, features_scores


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 800, 800)).resize(1, -1)
    print(dummy_input.shape)
    res = rpn(dummy_input)
    print(res[1].shape, res[2].shape)

# class FasterRCNN(nn.Module):
#     def __init__(self):
#         super(FasterRCNN, self).__init__()
#         self.
#
#     def forward(self, input):
