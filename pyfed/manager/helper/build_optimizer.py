import torch.optim as optim

from pyfed.optimizer.weight_perturbation import WpAdam
from pyfed.optimizer.fedprox import FedProx


def build_optimizer(config, trainable_params):
    if config.TRAIN_OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=config.TRAIN_LR,
            momentum=config.TRAIN_MOMENTUM,
            weight_decay=config.TRAIN_WEIGHT_DECAY
        )
    elif config.TRAIN_OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=config.TRAIN_LR,
            weight_decay=config.TRAIN_WEIGHT_DECAY)
    elif config.TRAIN_OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.TRAIN_LR,
            weight_decay=config.TRAIN_WEIGHT_DECAY)
    elif config.TRAIN_OPTIMIZER == 'wpadam':
        optimizer = WpAdam(
            trainable_params,
            lr=config.TRAIN_LR,
            alpha=config.TRAIN_WPADAM_ALPHA,
            weight_decay=config.TRAIN_WEIGHT_DECAY
        )
    elif config.TRAIN_OPTIMIZER == 'fedprox':
        optimizer = FedProx(
            trainable_params,
            lr=config.TRAIN_LR,
            weight_decay=config.TRAIN_WEIGHT_DECAY,
            mu=config.TRAIN_FEDPROX_MU
        )

    return optimizer
