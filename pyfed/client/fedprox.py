import torch

from .base import BaseClient


class FedProxClient(BaseClient):
    def __init__(self, config, site, server_model, partition=None):
        super(FedProxClient, self).__init__(config, site, server_model, partition)
        self.server_model = server_model

    def train(self, server_model=None):
        self.model.to(self.device)
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

            if step > 0:
                w_diff = torch.tensor(0., device=self.device)
                for w, w_t in zip(server_model.parameters(), self.model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

                w_diff = torch.sqrt(w_diff)
                loss += self.config.TRAIN_FEDPROX_MU / 2. * w_diff

            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)

            loss.backward()
            self.optimizer.step()

            self.curr_iter += 1

        loss = loss_all / len(self.train_loader)
        acc = self.metric_fn(outputs, labels)
        self.round += 1

        self.model.to('cpu')
        server_model.to('cpu')
        return loss, acc
