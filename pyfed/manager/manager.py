import os
import shutil

import numpy as np
import torch
import wandb

from pyfed.manager.comm import Comm
from pyfed.manager.helper.build_model import (
    build_model,
    build_client
)
from pyfed.utils.log import print_log


class Manager(object):
    def __init__(self, config):
        self.config = config
        self.best_acc = 0
        self.best_epoch = 0

        self._setup()

    def _setup(self):
        self.server_model = build_model(self.config)

        pytorch_total_params = sum(p.numel() for p in self.server_model.parameters())
        total_params_size = abs(pytorch_total_params * 4. / (1024 ** 2.))
        print('Network: {}, Params size (MB): {}'.format(self.config.NETWORK, total_params_size))

        self._build_clients()
        self.comm = Comm(self.server_model, self.config.COMM_TYPE)

    def _build_clients(self):
        client_class = build_client(self.config)
        print_log('Client type: {}'.format(self.config.CLIENT))

        if self.config.TRAIN_MODE == 'centralized':
            self.central = client_class(self.config, self.config.INNER_SITES, self.server_model)
        else:
            self.inner_clients = [client_class(self.config, site, self.server_model)
                                  for site in self.config.INNER_SITES]

            # if self.config.COMM_TYPE == 'FedAvg':
            #     self.train_nums = [len(client.train_loader.dataset) for client in self.inner_clients]
            #     total = sum(self.train_nums)
            #     self.client_weights = [num * 1.0 / total for num in self.train_nums]
            #     print(self.train_nums)
            #     print(self.client_weights)
            # else:
            self.client_weights = [1. / len(self.config.INNER_SITES) for _ in range(len(self.config.INNER_SITES))]

            if len(self.config.OUTER_SITES) > 0:
                self.outer_clients = [client_class(self.config, site, self.server_model)
                                      for site in self.config.OUTER_SITES]

    def train(self):
        metrics = {}
        best_avg_val_acc, best_avg_val_round = 0, 0
        best_server_val_acc, best_server_val_round = 0, 0
        best_person_val_acc, best_person_val_round = 0, 0

        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train(server_model=self.server_model)

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # client to server
            client_models = [client.client_to_server()['model'] for client in self.inner_clients]
            client_weights = self.client_weights

            self.server_model = self.comm(client_models, client_weights, self.server_model)

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_val_loss_' + client.name] = val_loss
                metrics['g_val_acc_' + client.name] = val_acc
                metrics['g_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_val_acc_avg'] = np.mean(val_accs)
            metrics['g_test_acc_avg'] = np.mean(test_accs)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            print_log('============== {} =============='.format('Local Validation'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val()
                test_loss, test_acc = client.test()
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['l_val_loss_' + client.name] = val_loss
                metrics['l_val_acc_' + client.name] = val_acc
                metrics['l_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['l_val_acc_avg'] = np.mean(val_accs)
            metrics['l_test_acc_avg'] = np.mean(test_accs)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            avg_val_acc = (metrics['g_val_acc_avg'] + metrics['l_val_acc_avg']) * 0.5
            if avg_val_acc > best_avg_val_acc:
                best_avg_val_acc = avg_val_acc
                best_avg_val_round = iter_round
            if metrics['g_val_acc_avg'] > best_server_val_acc:
                best_server_val_acc = metrics['g_val_acc_avg']
                best_server_val_round = iter_round
            if metrics['l_val_acc_avg'] > best_person_val_acc:
                best_person_val_acc = metrics['l_val_acc_avg']
                best_person_val_round = iter_round

            print_log('============== {} =============='.format('Summary'))
            print_log('best avg val round: {} | best avg val acc: {:.4f}'.format(best_avg_val_round, best_avg_val_acc))
            print_log('best server val round: {} | best server val acc: {:.4f}'.format(best_server_val_round, best_server_val_acc))
            print_log('best person val round: {} | best person val acc: {:.4f}'.format(best_person_val_round, best_person_val_acc))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)

            self.save(iter_round, np.mean(val_accs))

            wandb.log(metrics)

    def train_inner_outer(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    if self.config.CLIENT == 'fedprox' or self.config.CLIENT == 'fedproxcls':
                        train_loss, train_acc = client.train(server_model=self.server_model)
                    else:
                        train_loss, train_acc = client.train()

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # client to server
            client_models = [client.client_to_server()['model'] for client in self.inner_clients]
            client_weights = self.client_weights

            self.server_model = self.comm(client_models, client_weights, self.server_model, self.config.COMM_TYPE)

            # run global validation and testing
            print_log('============== {} =============='.format('Inner Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_inner_val_loss_' + client.name] = val_loss
                metrics['g_inner_val_acc_' + client.name] = val_acc
                metrics['g_inner_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_inner_val_acc_avg'] = np.mean(val_accs)
            metrics['g_inner_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[inner] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            print_log('============== {} =============='.format('Outer Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.outer_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_outer_val_loss_' + client.name] = val_loss
                metrics['g_outer_val_acc_' + client.name] = val_acc
                metrics['g_outer_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_outer_val_acc_avg'] = np.mean(val_accs)
            metrics['g_outer_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[outer] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)

            self.save(iter_round, np.mean(val_accs))

            wandb.log(metrics)

    def train_centralized(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                train_loss, train_acc = self.central.train()

                print_log('central| train loss: {:.4f} | train acc: {:.4f}'.format(train_loss, train_acc))

                metrics['train_loss_central'] = train_loss
                metrics['train_acc_central'] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_loss_list, val_acc_list = self.central.val()
            test_loss_list, test_acc_list = self.central.test()

            for site_idx, site in enumerate(self.config.INNER_SITES):
                metrics['g_val_loss_' + site] = val_loss_list[site_idx]
                metrics['g_val_acc_' + site] = val_acc_list[site_idx]
                metrics['g_test_acc_' + site] = test_acc_list[site_idx]

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    site, val_loss_list[site_idx], val_acc_list[site_idx], test_acc_list[site_idx]))

            metrics['l_val_acc_avg'] = np.mean(val_acc_list)
            metrics['l_test_acc_avg'] = np.mean(test_acc_list)
            print_log(
                'avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_acc_list), np.mean(test_acc_list)))

            self.save(iter_round, np.mean(val_acc_list))

            wandb.log(metrics)

    def train_individual(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train()

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val()
                test_loss, test_acc = client.test()
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['l_val_loss_' + client.name] = val_loss
                metrics['l_val_acc_' + client.name] = val_acc
                metrics['l_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['l_val_acc_avg'] = np.mean(val_accs)
            metrics['l_test_acc_avg'] = np.mean(test_accs)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            self.save(iter_round, np.mean(val_accs))

            wandb.log(metrics)

    def save(self, iter_round, val_acc):
        better = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = iter_round
            better = True

        # save server status
        save_dicts = {'server': self.server_model.state_dict(),
                      'best_epoch': self.best_epoch,
                      'best_acc': self.best_acc,
                      'round': iter_round}

        if self.config.TRAIN_MODE == 'centralized':
            save_dicts['optim'] = self.central.client_to_server()['optimizer'].state_dict()
        else:
            # save local status
            for ci, client in enumerate(self.inner_clients):
                save_dicts['optim_{}'.format(ci)] = client.client_to_server()['optimizer'].state_dict()

        torch.save(save_dicts, os.path.join(self.config.DIR_CKPT, 'model_latest.pth'))
        if better:
            shutil.copyfile(os.path.join(self.config.DIR_CKPT, 'model_latest.pth'),
                            os.path.join(self.config.DIR_CKPT, 'model_best.pth'))
