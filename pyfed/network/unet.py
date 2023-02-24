from collections import OrderedDict

import torch
import torch.nn as nn

from pyfed.utils.ffa_layer import FFALayer


def make_layers():
    layer1 = FFALayer(nfeat=32)
    layer2 = FFALayer(nfeat=64)
    layer3 = FFALayer(nfeat=128)
    layer4 = FFALayer(nfeat=256)
    layer5 = FFALayer(nfeat=512)
    return layer1, layer2, layer3, layer4, layer5


def _block(in_channels, features, name, affine=True, track_running_stats=True):
    bn_func = nn.BatchNorm2d

    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "_conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn1", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu1", nn.ReLU(inplace=True)),
                (
                    name + "_conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn2", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class UNetFedfa(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True):
        super(UNetFedfa, self).__init__()

        self.ffa_layer1, self.ffa_layer2, self.ffa_layer3, self.ffa_layer4, self.ffa_layer5 = make_layers()

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        enc1_ = self.ffa_layer1(enc1_)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        enc2_ = self.ffa_layer2(enc2_)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        enc3_ = self.ffa_layer3(enc3_)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        enc4_ = self.ffa_layer4(enc4_)

        bottleneck = self.bottleneck(enc4_)
        bottleneck = self.ffa_layer5(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1


if __name__ == '__main__':
    model = UNet(input_shape=(3, 384, 384))

    print(model.state_dict().keys())
