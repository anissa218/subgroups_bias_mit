import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

from models import basemodels
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from models.Regression.utils import regression_train, regression_val, regression_test
from models.utils import standard_train
from models.basenet import BaseNet


class Regression(BaseNet):
    def __init__(self, opt, wandb):
        super(Regression, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)
        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()
        
        # initialise model with pretrained weights - should not hard code but just testing for now
        # state_dict = torch.load('/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/your_path/fariness_data/model_records/UKBB_RET/Age/cusResNet18/baseline/all_filt2/42/42_best.pth')
        # print('loaded model from baseline/all_filt_2/42')
        # self.network.load_state_dict(state_dict['model'])


    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features= self.in_features, hidden_features = 1024).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
            
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        auc, train_loss,pred_df = regression_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb)

        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))

        # to distinguish between pretrained and not pretrained model results
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        self.epoch += 1
    
        
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        # meed to change standard val!
        auc, val_loss, log_dict, pred_df = regression_val(self.opt, self.network, loader, self._criterion, self.sens_classes, self.wandb)
        
        # anissa changes: save val pred_df
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_epoch_' + str(self.epoch)+'_val_pred.csv'), index = False)
        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_epoch_' + str(self.epoch)+'_val_pred.csv'), index = False)

        print('Validation epoch {}: validation loss:{}, AUC:{}'.format(
            self.epoch, val_loss, auc))
        return val_loss, auc, log_dict, pred_df
        
    def _test(self, loader):
        """Compute model output on testing set"""

        self.network.eval()
        # need to change standard test
        tol_output, tol_target, tol_sensitive, tol_index = regression_test(self.opt, self.network, loader, self._criterion, self.wandb)

        # should probably either remove these evals or change to regression type functions
        pred_df = pd.DataFrame(columns=['index', 'pred', 'label','raw_pred'])
        pred_df['index'] = tol_index
        pred_df['pred'] = tol_output
        pred_df['label'] = np.asarray(tol_target).squeeze()

        log_dict = {}

        # log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        # overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        # log_dict['Overall FPR'] = overall_FPR
        # log_dict['Overall FNR'] = overall_FNR
        
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_pred.csv'), index = False)
        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_pred.csv'), index = False)
        
        # for i, FPR in enumerate(FPRs):
        #     log_dict['FPR-group_' + str(i)] = FPR
        # for i, FNR in enumerate(FNRs):
        #     log_dict['FNR-group_' + str(i)] = FNR
        
        #log_dict = basics.add_dict_prefix(log_dict, 'Test ')
        
        return log_dict

    def record_val(self):
        best_pred_df = self.best_pred_df
        rmse = np.sqrt(((best_pred_df['label'].astype(float) - best_pred_df['pred'].astype(float))**2).mean())
        self.best_log_dict['RMSE'] = rmse
        
        # could do this for each group, attributes = self.val_meta

        log_dict = basics.add_dict_prefix(self.best_log_dict, 'Val ')
        print('Validation performance: ', log_dict)
        
        return pd.DataFrame(log_dict, index=[0])
