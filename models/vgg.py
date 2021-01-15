"""VGG for cifar dataset.
Ported form
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/vgg.py
"""
import torch.nn as nn

__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]


class VGG(nn.Module):
    def __init__(self, features, in_size=32, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, grayscale):
    """
    make VGG layer model

    Parameters
    ----------
    cfg: dict
        VGG configuration class ('A' 'B' 'D' 'E')
    grayscale: bool, default false
        True if gray scale input

    Returns
    -------
    model: nn.Sequential()
        VGG model layer with given configuration
    """
    layers = []
    in_dim = 1 if grayscale else 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_dim, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_dim = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11(**kwargs):
    model = VGG(make_layers(cfg["A"], grayscale=False), **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(make_layers(cfg["B"], grayscale=False), **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(make_layers(cfg["D"], grayscale=False), **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(make_layers(cfg["E"], grayscale=False), **kwargs)
    return model
