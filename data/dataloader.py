import torch
from torch.utils import data
import torch.nn.functional as F

import random


class DataLoader(data.DataLoader):
    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle


class BatchImageCollateFuncion(object):
    def __init__(
            self,
            scales=None,
            stop_epoch=None,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
        return images, targets

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1
