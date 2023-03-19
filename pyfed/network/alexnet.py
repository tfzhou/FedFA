from collections import OrderedDict

import torch
import torch.nn as nn

from pyfed.utils.ffa_layer import FFALayer


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True))
            ])
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True))
            ])
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True))
            ])
        )
        self.layer4 = nn.Sequential(
            OrderedDict([
                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True))
            ])
        )
        self.layer5 = nn.Sequential(
            OrderedDict([
                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True))
            ])
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 1024)),
                ('bn6', nn.BatchNorm1d(1024)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(1024, 1024)),
                ('bn7', nn.BatchNorm1d(1024)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(1024, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.classifier(f)

        return x


def make_layers():
    layer1 = FFALayer(nfeat=64)
    layer2 = FFALayer(nfeat=192)
    layer3 = FFALayer(nfeat=384)
    layer4 = FFALayer(nfeat=256)
    layer5 = FFALayer(nfeat=256)
    return layer1, layer2, layer3, layer4, layer5


class AlexNetFedFa(AlexNet):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, num_classes=10, **kwargs):
        super(AlexNetFedFa, self).__init__(num_classes, **kwargs)

        self.ffa_layer1, self.ffa_layer2, self.ffa_layer3, self.ffa_layer4, self.ffa_layer5 = make_layers()

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.ffa_layer1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.ffa_layer2(x)

        x = self.layer3(x)
        x = self.ffa_layer3(x)

        x = self.layer4(x)
        x = self.ffa_layer4(x)

        x = self.layer5(x)
        x = self.pool5(x)
        x = self.ffa_layer5(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = self.classifier(f)

        return x

    def forward_feat(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.ffa_layer1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.ffa_layer2(x)

        x = self.layer3(x)
        x = self.ffa_layer3(x)

        x = self.layer4(x)
        x = self.ffa_layer4(x)

        x = self.layer5(x)
        x = self.pool5(x)
        x = self.ffa_layer5(x)

        x = self.avgpool(x)

        return x
