import os
import torch


class BaseConfig:
    def __init__(self, server='euler', exp_name='fedavg'):
        self.EXP_NAME = exp_name

        assert server in ['euler', 'slurm']
        if server == 'euler':
            self.DIR_ROOT = os.environ.get('TMPDIR')
            self.DIR_DATA = os.path.join(self.DIR_ROOT, 'office_caltech_10')
            self.DIR_SAVE = os.path.join('/cluster/scratch/tiazhou/myresult/pyfed/office/', self.EXP_NAME)
            self.DIR_CKPT = os.path.join(self.DIR_SAVE, 'ckpt')
        elif server == 'slurm':
            self.DIR_ROOT = '/scratch_net/barbie_second/dataset/FL'
            self.DIR_DATA = os.path.join(self.DIR_ROOT, 'office_caltech_10')
            self.DIR_SAVE = os.path.join('/scratch_net/barbie_second/save/pyfed/office/', self.EXP_NAME)
            self.DIR_CKPT = os.path.join(self.DIR_SAVE, 'ckpt')

        self.DATASET = 'office'
        self.INNER_SITES = ['amazon', 'caltech', 'dslr', 'webcam']
        self.OUTER_SITES = []
        self.IMAGE_SIZE = [3, 256, 256]
        self.NUM_CLASSES = 10

        self.NETWORK = 'alexnet'
        self.NETWORK_PARAMS = {
            'num_classes': self.NUM_CLASSES
        }

        self.TRAIN_ROUNDS = 400
        self.TRAIN_EPOCH_PER_ROUND = 1

        self.TRAIN_LR = 1e-2
        self.TRAIN_BATCHSIZE = 32
        self.TRAIN_MOMENTUM = 0
        self.TRAIN_WEIGHT_DECAY = 0
        self.TRAIN_RESUME = False
        self.TRAIN_AUTO_RESUME = False
        self.TRAIN_GPU = 0
        self.TRAIN_OPTIMIZER = 'sgd'
        self.TRAIN_WARMUP_STEPS = 1000
        self.TRAIN_MIN_LR = 1e-5
        self.TRAIN_MODE = 'federated'  # ['individual', 'federated', 'centralized']
        self.TRAIN_LOSS = 'ce'
        self.TRAIN_RATIO = 0.6
        self.TEST_GPU = 0

        self.METRIC = 'top1'

        self.SEED = 0

        self.COMM_TYPE = 'FedAvg'
        self.CLIENT = 'BaseClient'

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('fedavg_prostate.py: cuda is not avalable')
        for path in [self.DIR_SAVE, self.DIR_CKPT]:
            if not os.path.isdir(path):
                os.makedirs(path)

