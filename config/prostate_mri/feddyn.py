from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, server='euler', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)

        self.CLIENT = 'FedDynClient'
        self.COMM_TYPE = 'FedDyn'
