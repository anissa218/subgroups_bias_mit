from models.utils import standard_train
from models.basenet import BaseNet
from importlib import import_module
import os


class baseline_simple(BaseNet):
    def __init__(self, opt, wandb):
        super(baseline_simple, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the network"""
        
        mod = import_module("models.basemodels")
        cusModel = getattr(mod, self.backbone)
        self.network = cusModel(n_classes=self.output_dim).to(self.device) # no need to speciy pretrained 
            
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        auc, train_loss,pred_df = standard_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb)

        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))

        # to distinguish between pretrained and not pretrained model results
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        self.epoch += 1
    