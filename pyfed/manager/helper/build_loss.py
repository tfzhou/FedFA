from monai.losses import DiceFocalLoss
import torch.nn as nn
from pyfed.utils.metric import Metric
from pyfed.loss.loss import JointLoss, DiceLoss


def build_loss(config):
    if config.TRAIN_LOSS == 'diceloss':
        return DiceLoss()
    if config.TRAIN_LOSS == 'ce':
        return nn.CrossEntropyLoss()
    if config.TRAIN_LOSS == 'dicefocal':
        return DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=True)
    return JointLoss()


def build_metric(config):
    return Metric(config.METRIC)

