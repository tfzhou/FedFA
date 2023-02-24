from .base import BaseClient


class FedBNClient(BaseClient):
    def __init__(self, config, site, server_model):
        super(FedBNClient, self).__init__(config, site, server_model)

    def server_to_client(self, server_model):
        for key in server_model.state_dict().keys():
            if 'bn' not in key:
                self.model.state_dict()[key].data.copy_(server_model.state_dict()[key])
