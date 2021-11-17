import numpy as np
from dataloader import get_dataloader
from collections import defaultdict
from progress.bar import IncrementalBar
import torch
import json

DIR_TO_SAVE_MODELS = 'models'
DIR_TO_SAVE_LOGS = 'logs'


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, df_train, df_val, batch_size):

        self.model = model
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size

        self.losses = {'train': [], 'val': []}

        self.dataloaders = get_dataloader(df_train, df_val, batch_size=self.batch_size)

        self.metrics = {'AP': accuracy_score, "BAP": balanced_accuracy_score}
        self.metrics_values = {
            phase: {name: [] for name in self.metrics.keys()} for phase in ['train', 'val']
        }

        if best_score is not None:
            self.best_score = best_score
        else:
            self.best_score = np.array([-np.inf for _ in self.metrics.keys()])

    def step(self, epoch, phase):
        epoch_loss = 0.0

        metrics = {name: [] for name in self.metrics.keys()}
        epoch_metric = {}

        # TODO: check syntax
        self.model.train() if phase == 'train' else self.model.eval()

        dataloader = self.dataloaders[phase]
        bar = IncrementalBar('Countdown', max=len(dataloader))

        self.optimizer.zero_grad()
        for i, (batch_images, batch_targets) in enumerate(dataloader):

            batch_images, batch_targets = torch.tensor(batch_images, dtype=torch.float).to(self.device),\
                                          torch.tensor(batch_targets, dtype=torch.long).to(self.device)

            with torch.set_grad_enabled(phase == 'train'):

                batch_predictions = self.model(batch_images)
                loss = self.criterion(batch_predictions, batch_targets)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Metric evaluating
                metrics = self.calculate_metrics(batch_targets, batch_predictions)

                epoch_loss += loss.item()
                bar.next()

            del batch_images, batch_targets, batch_predictions, loss

        bar.finish()

        epoch_loss = epoch_loss / len(dataloader)

        self.losses[phase].append(epoch_loss)

        for metric in self.metrics.keys():
            epoch_metric[metric] = np.mean(metrics[metric])
            self.metrics_values[phase][metric].append(epoch_metric[metric])

        torch.cuda.empty_cache()

        return epoch_loss, epoch_metric

    def calculate_metrics(self, batch_targets, batch_predictions) -> dict:
        metrics = {}
        for metric_name in self.metrics.keys():
            value = self.metrics[metric_name](
                batch_targets.cpu().detach().numpy(),
                batch_predictions.argmax(1).cpu().detach().numpy()
            )

            metrics[metric_name].append(value)

        return metrics

    def train(self, num_epochs):
        scores = None
        state = None
        bar = IncrementalBar('Countdown', max=len(num_epochs))

        for epoch in range(num_epochs):
            bar.next()

            loss, metric = self.step(epoch, 'train')

            print(f'Epoch {epoch} | train_loss {loss} | train_metric {metric}')

            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }

            loss, metric = self.step(epoch, 'val')

            print(f'Epoch {epoch} | val_loss {loss} | val_metric {metric}')

            self.scheduler.step(loss)

            scores = np.fromiter(metric.values(), dtype=np.float)
            if (scores > self.best_score).all():
                print('-' * 10 + 'New optimal model found and saved' + '-' * 10)
                state['best_metric'] = metric
                torch.save(state, "{}/model_epoch_{}_score_{:.4f}.pth".format(DIR_TO_SAVE_MODELS, epoch, scores[0]))
                self.best_score = scores

            losses_file = open("{}/losses/loss_epoch_{}.json".format(DIR_TO_SAVE_LOGS, epoch), "w")
            json.dump(self.losses, losses_file)
            losses_file.close()

            metrics_file = open("{}/metric/metric_epoch_{}.json".format(DIR_TO_SAVE_LOGS, epoch), "w")
            json.dump(self.metrics_values, metrics_file)
            metrics_file.close()

        bar.finish()
        # saving last epoch
        print('-' * 10 + str(num_epochs) + 'passed' + '-' * 10)
        torch.save(state, "{}/model_epoch_{}_score_{:.4f}.pth".format(DIR_TO_SAVE_MODELS, num_epochs, scores[0]))
