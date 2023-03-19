from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, server='euler', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)
        self.CLIENT = 'BaseClient'
        self.COMM_TYPE = 'FedFA'
        self.NETWORK = 'alexnetfedfa'
        self.WITH_ALIGN = True

        self.NETWORK_PARAMS = {
            'num_classes': 10,
        }
