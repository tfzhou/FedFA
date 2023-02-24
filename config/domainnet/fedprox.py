from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, server='euler', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)
        self.CLIENT = 'FedProxClient'
        self.COMM_TYPE = 'FedAvg'
        self.TRAIN_FEDPROX_MU = 1e-2

        self.NETWORK_PARAMS = {'num_classes': 10}

