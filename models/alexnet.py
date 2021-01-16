"""Alexnet for cifar dataset.
Ported form
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/alexnet.py
"""

import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, in_size=32, num_classes=10, grayscale=False):
        """
        Constructs a AlexNet model.

        Parameters
        ----------
        in_size: int, default 32
            Input image size
        num_classes: int, default 10
            Num of output classes
        grayscale: bool, default false
            True if gray scale input

        Returns
        -------
        model: AlexNet model class
            AlexNet model class with given parameters
        """
        super(AlexNet, self).__init__()
        in_dim = 1 if grayscale else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model
