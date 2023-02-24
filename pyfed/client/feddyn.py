import copy

import torch

from .base import BaseClient


class FedDynClient(BaseClient):
    def __init__(self, config, site, server_model, partition=None):
        super(FedDynClient, self).__init__(config, site, server_model, partition)
        self.server_model = server_model
        self.gradL = copy.deepcopy(server_model)
        for key in self.gradL.state_dict().keys():
            self.gradL.state_dict()[key].data.copy_(torch.zeros_like(self.gradL.state_dict()[key]).float())
        self.gradL.to(self.device)
        self.alpha = 0.1

    def train(self, server_model=None):
        self.model.to(self.device)

        src_model = copy.deepcopy(self.model)
        src_model.eval()

        self.model.train()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        server_model.to(self.device)

        for step, (image, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.loss_fn(output, label)
            loss2, loss3 = 0, 0

            for pgl, pm, ps in zip(self.gradL.parameters(), self.model.parameters(), src_model.parameters()):
                loss2 += torch.dot(pgl.view(-1), pm.view(-1))
                loss3 += torch.sum(torch.pow(pm - ps, 2))
            loss = loss - loss2 + 0.5 * self.alpha * loss3

            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)

            loss.backward()
            self.optimizer.step()

            self.curr_iter += 1

        for key in self.gradL.state_dict().keys():
            self.gradL.state_dict()[key].data.copy_(
                self.gradL.state_dict()[key] - self.alpha * (self.model.state_dict()[key] - src_model.state_dict()[key]))

        loss = loss_all / len(self.train_loader)
        acc = self.metric_fn(outputs, labels)
        self.round += 1

        self.model.to('cpu')
        server_model.to('cpu')
        return loss, acc
