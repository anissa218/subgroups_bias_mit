from models.baseline import baseline


class subsampling(baseline):
    def __init__(self, opt, wandb):
        super(subsampling, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
    