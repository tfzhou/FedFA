import torch

from pyfed.optimizer.sam import SAM
from pyfed.utils.bypass_bn import enable_running_stats, disable_running_stats
from .base import BaseClient


class FedSAMClient(BaseClient):
    def __init__(self, config, site, server_model, partition=None):
        super(FedSAMClient, self).__init__(config, site, server_model, partition)

        self.optimizer = SAM(
            self.model.parameters(),
            base_optimizer=torch.optim.Adam,
            lr=config.TRAIN_LR,
            weight_decay=config.TRAIN_WEIGHT_DECAY
        )

    def train(self, server_model=None):
        self.model.to(self.device)
        self.model.train()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        for step, (image, label) in enumerate(self.train_loader):
            image, label = image.to(self.device), label.to(self.device)

            # first forward-backward step
            enable_running_stats(self.model)
            output = self.model(image)
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.model)
            second_output = self.model(image)
            self.loss_fn(second_output, label).backward()
            self.optimizer.second_step(zero_grad=True)

            loss_all += loss.item()

            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)
            self.curr_iter += 1

        loss = loss_all / len(self.train_loader)
        acc = self.metric_fn(outputs, labels)
        self.round += 1

        self.model.to('cpu')
        return loss, acc