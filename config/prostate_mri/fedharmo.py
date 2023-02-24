from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, server='euler', exp_name='fedavg'):
        super(Config, self).__init__(server, exp_name)

        self.CLIENT = 'harmo_fl'
        self.TRAIN_OPTIMIZER = 'wpadam'
        self.TRAIN_WPADAM_ALPHA = 0.05
        self.COMM_TYPE = 'average'
        self.AUG_METHOD = None

        self.NETWORK = 'unetharmo'
        self.NETWORK_PARAMS = {
            'input_shape': self.IMAGE_SIZE
        }


