import torch

from .base import BaseClient


class FedHarmoClient(BaseClient):
    def __init__(self, config, site, server_model):
        super(FedHarmoClient, self).__init__(config, site, server_model)

    def train(self, server_model=None):
        self.model.to(self.device)
        self.model.train()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        for step, (image, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.loss_fn(output, label)
            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)

            loss.backward()
            self.optimizer.generate_delta(zero_grad=True)
            self.loss_fn(self.model(image), label).backward()
            self.optimizer.step(zero_grad=True)

        loss = loss_all / len(self.train_loader)
        acc = self.metric_fn(outputs, labels)

        self.model.to('cpu')
        return loss, acc

    def server_to_client(self, server_model):
        for key in server_model.state_dict().keys():
            self.model.state_dict()[key].data.copy_(server_model.state_dict()[key])
            if 'running_amp' in key:
                self.model.amp_norm.fix_amp = True


