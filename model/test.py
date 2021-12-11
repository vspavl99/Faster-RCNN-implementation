import cv2
import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


def draw(proposals, ground_true_bboxes):
    import torch
    import cv2
    import numpy as np
    dummy_image = np.array(torch.ones(224, 224, 3, dtype=torch.uint8))

    for i, anchor in enumerate(proposals[0]):
        x1, y1, x2, y2 = int(anchor[0].item()), int(anchor[1].item()), int(anchor[2].item()), int(anchor[3].item())
        # print(x1, y1, x2, y2)

        dummy_image = cv2.rectangle(img=dummy_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255))
        # break
    for i, anchor in enumerate(ground_true_bboxes[0]):
        x1, y1, x2, y2 = int(anchor[0].item()), int(anchor[1].item()), int(anchor[2].item()), int(anchor[3].item())
        # print(x1, y1, x2, y2)

        dummy_image = cv2.rectangle(img=dummy_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0))
        # break

    import matplotlib.pyplot as plt
    plt.imshow(dummy_image)
    plt.show()


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3

    model = vgg16(pretrained=False)

    features = list(model.features)[:30]
    classifier = model.classifier

    # (classifier): Sequential(
    # (0): Linear(in_features=25088, out_features=4096, bias=True)
    # (1): ReLU(inplace=True)
    # (2): Dropout(p=0.5, inplace=False)
    # (3): Linear(in_features=4096, out_features=4096, bias=True)
    # (4): ReLU(inplace=True)
    # (5): Dropout(p=0.5, inplace=False)
    # (6): Linear(in_features=4096, out_features=1000, bias=True)

    classifier = list(classifier)
    del classifier[6]
    # if False:
    #     del classifier[5]
    #     del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


def backbone(x):
    model = vgg16(pretrained=True).features
    model.eval()

    return model(x)


def read_image():
    import cv2
    image = cv2.imread(r'..\data\PASCAL VOC 2012\JPEGImages\2007_000032.jpg')
    assert image is not None
    image = torch.tensor(image)
    image = image.permute(2, 0, 1).unsqueeze(0).divide(255)
    print(image.shape, )
    return image


def train_rpn():
    from model.rpn import RPN

    rpn = RPN()

    optimizer = torch.optim.Adam(rpn.parameters())

    for i in range(100):
        dummy_input = torch.ones((2, 3, 224, 224))

        feature_map, proposals, loss = rpn(
            dummy_input, torch.tensor([[[50, 50, 150, 150]], [[50, 50, 150, 150]]])
        )
        print(loss)
        loss = sum(value for key, value in loss.items())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)
        if i % 10 == 0:
            draw(proposals, torch.tensor([[[50, 50, 150, 150]]]))


if __name__ == '__main__':
    # anchor_generator = AnchorGenerator(
    #     sizes=((32, 64),),
    #     aspect_ratios=(1., 0.5)
    # )
    #
    # # anchor_generator
    # dummy_image = read_image()
    # output = backbone(dummy_image)
    # print(output.shape)
    #
    # anchors = anchor_generator(
    #     ImageList(dummy_image, [image.shape for image in dummy_image]),
    #     output)
    #
    # image = (dummy_image.squeeze(0).permute(1, 2, 0) * 255).type(torch.uint8).numpy()
    # for i, anchor in enumerate(anchors[0]):
    #     x1, y1, x2, y2 = int(anchor[0].item()), int(anchor[1].item()), int(anchor[2].item()), int(anchor[3].item())
    #     print(x1, y1, x2, y2)
    #
    #     image = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255))
    #     # break
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    train_rpn()