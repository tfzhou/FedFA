from pyfed.manager.helper.build_dataset import build_central_dataset


class Central(object):
    def __init__(self, config, sites, server_model):
        super(Central, self).__init__(config, sites, server_model)
        self._setup()

    def _setup(self):
        self.train_loader, self.valid_loaders, self.test_loaders = \
            build_central_dataset(self.config, self.sites)
