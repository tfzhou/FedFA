import argparse
import importlib
import random
import numpy as np
import wandb

import torch
import torch.backends.cudnn as cudnn

from pyfed.manager.manager import Manager


def cli_main(config):
    manager = Manager(config)

    print('mode: ', config.TRAIN_MODE)

    if config.TRAIN_MODE == 'federated':
        print('train')
        manager.train()
    elif config.TRAIN_MODE == 'individual':
        manager.train_individual()
    elif config.TRAIN_MODE == 'centralized':
        manager.train_centralized()
    elif config.TRAIN_MODE == 'innerouter':
        manager.train_inner_outer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='fedavg')
    parser.add_argument('--inner_sites', nargs='+', default=[])
    parser.add_argument('--outer_sites', nargs='+', default=[])
    parser.add_argument('--server', default='euler')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--trial', default=0)
    parser.add_argument('--train_ratio', default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run_name', required=True, help='A short display name for this run, which is how you will '
                                                          'identify this run in the UI.')
    parser.add_argument('--run_notes', required=True, help='A longer description of the run, like a -m commit message '
                                                           'in git.')

    args = parser.parse_args()
    print('server:', args.server)

    config_cls = getattr(importlib.import_module(args.config), 'Config')
    config = config_cls(args.server, args.exp_name + '_trial{}'.format(args.trial))

    if args.seed is not None:
        config.SEED = args.seed

    print('seed:', config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('NETWORK_PARAMS:', config.NETWORK_PARAMS)

    if len(args.inner_sites) > 0:
        config.INNER_SITES = args.inner_sites
    if len(args.outer_sites) > 0:
        config.OUTER_SITES = args.outer_sites
    if args.train_ratio is not None:
        config.TRAIN_RATIO = float(args.train_ratio)

    print(args)
    wandb.init(config=config, project=config.EXP_NAME,
               name=args.run_name, notes=args.run_notes,
               settings=wandb.Settings(start_method="fork"))

    cli_main(config)
