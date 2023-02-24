import random
import numpy as np

import torch
import torchvision.transforms as transforms
import pyfed.utils.hecktor_transforms as hecktor_transforms
from pyfed.dataset.dataset import Prostate, Nuclei, DomainNetDataset, HecktorDataset, OfficeDataset, VLCSDataset
from pyfed.utils.log import print_log


def build_dataset(config, site):
    assert site in config.INNER_SITES + config.OUTER_SITES
    if config.DATASET == 'prostate':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = Prostate(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                             split='train', transform=transform)
        valid_set = Prostate(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                             split='valid', transform=transform)
        test_set = Prostate(site=site, base_path=config.DIR_DATA,train_ratio=config.TRAIN_RATIO,
                            split='test', transform=transform)
    elif config.DATASET == 'hecktor':
        train_transform = hecktor_transforms.Compose([
            hecktor_transforms.RandomRotation(p=0.5, angle_range=[0, 30]),
            hecktor_transforms.Mirroring(p=0.5),
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        valid_transform = hecktor_transforms.Compose([
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        train_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='train', transforms=train_transform)
        valid_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='valid', transforms=valid_transform)
        test_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='test', transforms=valid_transform)
    elif config.DATASET == 'nuclei':
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
        ])

        train_set = Nuclei(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
        valid_set = Nuclei(site=site, base_path=config.DIR_DATA, split='val', transform=transform)
        test_set = Nuclei(site=site, base_path=config.DIR_DATA, split='test', transform=transform)
    elif config.DATASET == 'domainnet':
        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='train', transform=train_transform)
        valid_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='val', transform=test_transform)
        test_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                    split='test', transform=test_transform)
    elif config.DATASET == 'office':
        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='train', transform=train_transform)
        valid_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='val', transform=test_transform)
        test_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                    split='test', transform=test_transform)

    elif config.DATASET == 'vlcs':
        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                  split='train', transform=train_transform)
        valid_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                  split='val', transform=test_transform)
        test_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                 split='test', transform=test_transform)


    print_log('[Client {}] Train={}, Val={}, Test={}'.format(site, len(train_set), len(valid_set), len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE,
                                               shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE,
                                               shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE,
                                              shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader


def build_central_dataset(config, sites):
    train_sets, valid_sets, test_sets = [], [], []
    train_loaders, valid_loaders, test_loaders = [], [], []
    if config.DATASET == 'prostate':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for site in sites:
            train_set = Prostate(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
            valid_set = Prostate(site=site, base_path=config.DIR_DATA, split='valid', transform=transform)
            test_set = Prostate(site=site, base_path=config.DIR_DATA, split='test', transform=transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    elif config.DATASET == 'hecktor':
        train_transform = hecktor_transforms.Compose([
            hecktor_transforms.RandomRotation(p=0.5, angle_range=[0, 30]),
            hecktor_transforms.Mirroring(p=0.5),
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        valid_transform = hecktor_transforms.Compose([
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        for site in sites:
            train_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='train', transforms=train_transform)
            valid_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='valid', transforms=valid_transform)
            test_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='test', transforms=valid_transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(
                torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    elif config.DATASET == 'nuclei':
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
        ])

        for site in sites:
            train_set = Nuclei(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
            valid_set = Nuclei(site=site, base_path=config.DIR_DATA, split='val', transform=transform)
            test_set = Nuclei(site=site, base_path=config.DIR_DATA, split='test', transform=transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(
                torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    return train_loader, valid_loaders, test_loaders