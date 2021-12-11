from progress.bar import IncrementalBar

from torch.optim import Adam
from train.dataloader import ConfigDataset, get_dataloader, DatasetFasterRCNN
from pathlib import Path
from evaluate.engine import evaluate

import json
import torch

DIR_TO_SAVE_MODELS = '/home/user/test_network'
DIR_TO_SAVE_LOGS = '/home/user/test_network'


class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, dataloaders):

        self.model = model
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.device = device

        self.losses = {'train': [], 'val': []}
        self.dataloaders = dataloaders

        # self.metrics = {'AP': accuracy_score, "BAP": balanced_accuracy_score}
        # self.metrics_values = {
        #     phase: {name: [] for name in self.metrics.keys()} for phase in ['train', 'val']
        # }
        # if best_score is not None:
        #     self.best_score = best_score
        # else:
        #     self.best_score = np.array([-np.inf for _ in self.metrics.keys()])

    def step(self, phase):
        epoch_loss = 0.0

        self.model.train() if phase == 'train' else self.model.eval()

        dataloader = self.dataloaders[phase]
        bar = IncrementalBar('Progress bar', max=len(dataloader))

        self.optimizer.zero_grad()
        for i, (batch_images, batch_targets) in enumerate(dataloader):

            batch_images = torch.stack(batch_images, dim=0).to(self.device) / 255
            batch_targets = [
                {
                    'boxes': torch.Tensor(targets['bboxes']).to(self.device),
                    'labels': targets['labels'].type(torch.int64).to(self.device)
                }
                # if len(data['bboxes']) != 0 else
                # {
                #     'boxes': torch.zeros(1, 4).to(self.device),
                #     'labels': torch.zeros(1).type(torch.int64).to(self.device)
                # }
                for targets in batch_targets
            ]

            with torch.set_grad_enabled(phase == 'train'):

                output = self.model(batch_images, batch_targets)
                # batch_predictions, loss = self.model(batch_images, batch_targets)

                if phase == 'train':
                    if isinstance(output, tuple):
                        predictions, output = output

                    loss = sum(value for key, value in output.items())
                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()

                bar.next()

            if i % 10 == 0:
                print(loss)

            # del batch_images, batch_targets, batch_predictions, loss
            del batch_images, batch_targets, loss

        bar.finish()

        epoch_loss = epoch_loss / len(dataloader)
        self.losses[phase].append(epoch_loss)
        torch.cuda.empty_cache()

        return epoch_loss, None

    def train(self, num_epochs):
        state = None
        bar = IncrementalBar('Countdown', max=num_epochs)

        for epoch in range(num_epochs):
            bar.next()

            loss, metric = self.step('train')

            print(f'Epoch {epoch} | train_loss {loss} | train_metric {metric}')

            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }

            evaluate(self.model, self.dataloaders['val'], device=self.device)
            torch.save(state, "{}/torch_model_epoch_{}.pth".format(DIR_TO_SAVE_MODELS, epoch))
            self.scheduler.step(loss)

        bar.finish()
        # saving last epoch
        print('-' * 10 + str(num_epochs) + 'passed' + '-' * 10)
        torch.save(state, "{}/net_model_epoch_{}_score.pth".format(DIR_TO_SAVE_MODELS, num_epochs))


# def torch_faster():
# pass
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = torchvision.models.vgg16(pretrained=True).features

backbone.out_channels = 512
anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                   aspect_ratios=((0.5, 1.0,),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
faster_rcnn = FasterRCNN(backbone,
                   num_classes=21,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)


# from model.FasterRCNN import FasterRCNN


if __name__ == '__main__':

    dataloaders = get_dataloader(
        path_train_csv=Path('../data/annotation_train.csv'),
        path_val_csv=Path('../data/annotation_val.csv'),
        shuffle=False, batch_size=20
    )

    # faster_rcnn = FasterRCNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    faster_rcnn.to(device)

    optimizer = Adam(faster_rcnn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode="min", patience=3, verbose=True)

    model_train = ModelTrainer(faster_rcnn, optimizer, scheduler, device, dataloaders)
    model_train.train(30)

    # faster_rcnn.load_state_dict(torch.load('/home/user/test_network/net_model_epoch_15_score.pth')['state_dict'])
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # faster_rcnn.to(device)
    # faster_rcnn.eval()
    # out = faster_rcnn(torch.ones(1, 3, 800, 800).to(device))

    # faster_rcnn = FasterRCNN()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # faster_rcnn.to(device)
    #
    # evaluate(faster_rcnn, dataloaders['val'], device=device)
