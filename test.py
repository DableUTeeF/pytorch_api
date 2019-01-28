import models
import torch.nn as nn
import torch
import numpy as np


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.output(x)
        return torch.sigmoid(x)


class GenTest:
    def __next__(self):
        x = np.random.rand(32, 10)
        y = np.random.rand(32)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

    def __len__(self):
        return 100

    def __iter__(self):
        for x in range(len(self)):
            yield self.__next__()


if __name__ == '__main__':
    model = models.Model(M())
    model.compile('sgd', nn.BCELoss(), 'acc')
    h = model.fit_generator(GenTest(), 1, validation_data=GenTest())
    print(h)
