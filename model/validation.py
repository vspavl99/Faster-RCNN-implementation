import torch
from train.dataloader import get_dataloader
from pathlib import Path
import cv2
import numpy as np
# from model.FasterRCNN import FasterRCNN


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
backbone = torchvision.models.vgg16(pretrained=True).features
backbone.out_channels = 512
anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                   aspect_ratios=((0.5, 1.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=21,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

import matplotlib.pyplot as plt
from torchvision.ops import batched_nms, nms

if __name__ == '__main__':
    # model = FasterRCNN()
    model.load_state_dict(torch.load('/home/user/test_network/net_model_epoch_15_score.pth')['state_dict'])
    model.eval()

    dataloaders = get_dataloader(
        path_train_csv=Path('../data/annotation_val.csv'),
        path_val_csv=Path('../data/annotation_val.csv'),
        shuffle=True, batch_size=4
    )
    classes = (
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    for k,  (batch_images, batch_targets) in enumerate(dataloaders['val']):

        batch_images = torch.stack(batch_images, dim=0).to(device) / 255
        # batch_output = model(batch_images)[0]
        batch_output = model(batch_images)
        for j, output in enumerate(batch_output):
            image = batch_images[j].detach().cpu().permute(1, 2, 0) * 255
            image = image.type(torch.uint8).numpy()
            image = np.array(image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            bboxes = output['boxes'].detach().cpu()
            labels = output['labels'].detach().cpu().type(bboxes.dtype)
            scores = output['scores'].detach().cpu().type(bboxes.dtype)
            keep = nms(bboxes, scores, 0.3)

            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            keep = torch.where(scores > 0.7)[0]
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for box, label, score in zip(bboxes, labels, scores):
                x1, y1, x2, y2 = int(box[0]) + 1, int(box[1]) + 1, int(box[2]) + 1, int(box[3]) + 1
                image = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=2)
                image = cv2.putText(image, f"{classes[int(label)]} {score:.4}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            plt.imshow(image)
            plt.show()

        del batch_images, batch_output