from pyfed.network.unet import UNet, UNetFedfa
from pyfed.network.alexnet import AlexNet, AlexNetFedFa

from pyfed.client import (
    BaseClient,
    FedProxClient,
    FedHarmoClient,
    FedBNClient,
    FedSAMClient,
    FedDynClient,
)


def build_model(config):
    if config.NETWORK == 'unet':
        model = UNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unetfedfa':
        model = UNetFedfa(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'alexnet':
        model = AlexNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'alexnetfedfa':
        model = AlexNetFedFa(**config.NETWORK_PARAMS)

    return model


def build_client(config):
    assert config.CLIENT in [
        'BaseClient',
        'FedProxClient',
        'FedHarmoClient',
    ]
    client_class = eval(config.CLIENT)

    return client_class