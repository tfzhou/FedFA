import torch
import copy


class FedAvgComm:
    def __init__(self, server_model=None):
        pass

    def __call__(self, clients, weights, server):
        return average(clients, weights, server)


class FedAvgMComm:
    def __init__(self, server_model=None):
        self.beta = 0.01
        self.v = copy.deepcopy(server_model)
        for key in self.v.state_dict().keys():
            self.v.state_dict()[key].data.copy_(torch.zeros_like(self.v.state_dict()[key]).float())

    def __call__(self, clients, weights, server):
        server_copy = copy.deepcopy(server)
        server = average(clients, weights, server)

        for key in server.state_dict().keys():
            self.v.state_dict()[key].data.copy_(self.beta * self.v.state_dict()[key] +
                                                server_copy.state_dict()[key] - server.state_dict()[key])
            server.state_dict()[key].data.copy_(server_copy.state_dict()[key] - self.v.state_dict()[key])

        return server


class FedDynComm:
    def __init__(self, server_model=None):
        self.alpha = 0.1
        self.v = copy.deepcopy(server_model)
        for key in self.v.state_dict().keys():
            self.v.state_dict()[key].data.copy_(torch.zeros_like(self.v.state_dict()[key]).float())

    def __call__(self, clients, weights, server):
        server_copy = copy.deepcopy(server)
        server = average(clients, weights, server)

        for key in server.state_dict().keys():
            tmp = torch.zeros_like(server.state_dict()[key]).float()
            for client_idx in range(len(weights)):
                tmp += weights[client_idx] * clients[client_idx].state_dict()[key]

            v = self.v.state_dict()[key] - self.alpha * (tmp - server_copy.state_dict()[key])
            self.v.state_dict()[key].data.copy_(v)

            server.state_dict()[key].data.copy_(server.state_dict()[key] - 1.0 / self.alpha * v)

        return server


class FedFAComm:
    def __init__(self, server_model=None):
        pass

    def __call__(self, clients, weights, server):
        for key in server.state_dict().keys():
            tmp = torch.zeros_like(server.state_dict()[key]).float()
            for client_idx in range(len(weights)):
                tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
            server.state_dict()[key].data.copy_(tmp)

            if 'running_var_mean_bmic' in key or 'running_var_std_bmic' in key:
                tmp = []
                for client_idx in range(len(weights)):
                    tmp.append(clients[client_idx].state_dict()[key.replace('running_var_', 'running_')])

                tmp = torch.stack(tmp)
                var = torch.var(tmp)
                server.state_dict()[key].data.copy_(var)

                # wandb.log({'server.{}'.format(key): torch.norm(var).cpu().numpy()}, commit=False)

        return server


class FedBNComm:
    def __init__(self, server_model=None):
        pass

    def __call__(self, clients, weights, server):
        for key in server.state_dict().keys():
            if 'bn' not in key:
                tmp = torch.zeros_like(server.state_dict()[key])
                for client_idx in range(len(weights)):
                    tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
                server.state_dict()[key].data.copy_(tmp)

        return server


def average(clients, weights, server):
    for key in server.state_dict().keys():
        tmp = torch.zeros_like(server.state_dict()[key]).float()
        for client_idx in range(len(weights)):
            tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
        server.state_dict()[key].data.copy_(tmp)

    return server


class Comm:
    def __init__(self, server_model=None, comm_type='FedAvg'):
        if comm_type == 'FedAvg':
            self.comm_fn = FedAvgComm()
        elif comm_type == 'FedAvgM':
            self.comm_fn = FedAvgMComm(server_model)
        elif comm_type == 'FedDyn':
            self.comm_fn = FedDynComm(server_model)
        elif comm_type == 'FedFA':
            self.comm_fn = FedFAComm()
        elif comm_type == 'FedBN':
            self.comm_fn = FedBNComm()

    def __call__(self, clients, weights, server):
        return self.comm_fn(clients, weights, server)
