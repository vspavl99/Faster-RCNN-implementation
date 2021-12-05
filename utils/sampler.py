import torch
from torch import Tensor


class Sampler:
    def __init__(self, mini_batch_size: int = 256):
        self.mini_batch_size = mini_batch_size

    def create_minibatch(self, batch_labels: list[Tensor]) -> tuple[list[Tensor], list[Tensor], ...]:
        """
         Take mini_batch_size elements from labels, and create minibatch of them
        :param batch_labels: list of Tensor with -1, 0 or index of gt_box (positive value)
        :return:
        """
        batch_positive_samples, batch_negative_samples = [], []
        batch_samples = []

        for labels in batch_labels:
            positive = torch.where(labels >= 1)[0]
            negative = torch.where(labels == 0)[0]

            number_of_positive = min(len(positive), self.mini_batch_size // 2)
            number_of_negative = min(self.mini_batch_size - number_of_positive, len(negative))

            positive_indexes = torch.randperm(len(positive), device=positive.device)[:number_of_positive]
            negative_indexes = torch.randperm(len(negative), device=positive.device)[:number_of_negative]

            positive_indexes = positive[positive_indexes]
            negative_indexes = negative[negative_indexes]

            positive_sample = torch.zeros_like(labels, device=labels.device, dtype=torch.uint8)
            positive_sample[positive_indexes] = 1
            batch_positive_samples.append(positive_sample)

            negative_sample = torch.zeros_like(labels, device=labels.device, dtype=torch.uint8)
            negative_sample[negative_indexes] = 1
            batch_negative_samples.append(negative_sample)

            # TODO: choose implementation
            #######
            mini_batch_samples = torch.where(negative_sample | positive_sample)[0]
            batch_samples.append(mini_batch_samples)
            ######

        return batch_positive_samples, batch_negative_samples, batch_samples
