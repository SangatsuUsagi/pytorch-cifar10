'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
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
            nn.Conv2d(in_dim, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model
