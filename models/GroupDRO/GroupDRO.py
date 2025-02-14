import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import basemodels
from utils import basics
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR, get_pred_df
from models.basenet import BaseNet
from importlib import import_module
from models.GroupDRO.utils import LossComputer
    
    
class GroupDRO(BaseNet):
    def __init__(self, opt, wandb):
        super(GroupDRO, self).__init__(opt, wandb)
        
        self.set_network(opt)
        self.set_optimizer(opt)
        
        self.groupdro_alpha = opt['groupdro_alpha']
        self.groupdro_gamma = opt['groupdro_gamma']
        self.groupdro_step = opt['groupdro_step']
        self.adj = opt['groupdro_adj']
        self.groupdro_step = opt['groupdro_step']
        self.pure = opt['groupdro_pure']
        self.use_train_loss_for_val = opt['groupdro_use_train_loss_for_val']
        self.pure_val = opt['groupdro_pure_val_loss']

        self.register_buffer("q", torch.ones(self.sens_classes))
        
        self.criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
        adjustments = [self.adj]
        assert len(adjustments) in (1, self.train_data.sens_classes)
        if len(adjustments)==1:
            adjustments = np.array(adjustments* self.train_data.sens_classes)
        else:
            adjustments = np.array(adjustments)
        if self.groupdro_alpha != 1:
            btl=True
        else:
            btl=False
        self.train_loss_computer = LossComputer(
            criterion = self._criterion,
            is_robust=True,
            dataset=self.train_data,
            alpha=self.groupdro_alpha,
            gamma=self.groupdro_gamma,
            adj=adjustments,
            step_size=self.groupdro_step,
            normalize_loss=False,
            btl=btl,
            min_var_weight=0,
            pure = self.pure)
    
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
        
        running_loss, auc = 0., 0.
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        no_iter = 0
        dro_results = {}
        group_losses, mean_losses, losses,avg_group_losses,worst_train_indices,list_of_weights = [], [], [], [],[],[]
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)
            self.optimizer.zero_grad()
            outputs, features = self.network.forward(images)
            
            group_loss,mean_loss,loss,worst_index,weights = self.train_loss_computer.loss(outputs, targets, sensitive_attr, is_training = True)
            # group loss is list of losses for each group, mean loss is avg for each sample, loss is robust DRO loss, index is index of worst group
            avg_group_loss = self.train_loss_computer.avg_group_loss
            group_losses.append(group_loss)
            mean_losses.append(mean_loss)
            losses.append(loss)
            avg_group_losses.append(avg_group_loss)
            running_loss += loss.item()
            worst_train_indices.append(worst_index)
            list_of_weights.append(weights)
            
            loss.backward()
                    
            self.optimizer.step()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                try:
                    self.wandb.log({'Training loss': running_loss / (i+1), 'Training AUC': auc / (i+1)})
                except:
                    pass
            
            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()

        dro_results['group_losses'] = group_losses
        dro_results['mean_losses'] = mean_losses
        dro_results['losses'] = losses
        dro_results['avg_group_losses'] = avg_group_losses
        dro_results['train_indices'] = worst_train_indices
        dro_results['weights'] = list_of_weights
        # save dict
        torch.save(dro_results, os.path.join(self.save_path, 'dro_loss_epoch_' + str(self.epoch) + '.pth'))
        
        self.scheduler.step()
        
        pred_df = get_pred_df(tol_output, tol_target, tol_sensitive, tol_index)
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)
        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)
       

        running_loss /= no_iter
        auc = auc / no_iter
        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, running_loss))
        self.epoch += 1
        
    def _val(self, loader):
        """Compute model output on validation set"""

        self.network.eval()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        val_list_of_weights = []
        val_dro_results = {}
        val_loss, auc = 0., 0.
        no_iter = 0
        # val loss computer re-initialises at every epoch
        self.val_loss_computer = LossComputer(
            criterion = self._criterion,
            is_robust=True,
            dataset=self.val_data,
            alpha=self.groupdro_alpha,
            gamma=self.groupdro_gamma,
            step_size=self.groupdro_step,
            normalize_loss=False,
            btl=False, # have set it to always false (same as original implementation)=
            min_var_weight=0,
            pure = self.pure_val) # copying gDRO original implementation. not exactly same params as train_loss_computer
        if self.pure_val:
            print('using pure val loss')

        group_losses, mean_losses, losses,avg_group_losses,worst_train_indices,list_of_weights = [], [], [], [],[],[]

        with torch.no_grad():
            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                if self.use_train_loss_for_val:
                    print('using train loss weights for val')
                    group_loss,mean_loss,loss,worst_index,weights = self.train_loss_computer.loss(outputs, targets, sensitive_attr, is_training = False)
                    avg_group_loss = self.train_loss_computer.avg_group_loss
                else: # this should be default!!!
                    group_loss,mean_loss,loss,worst_index,weights = self.val_loss_computer.loss(outputs, targets, sensitive_attr, is_training = False)
                    avg_group_loss = self.val_loss_computer.avg_group_loss

                val_loss += loss.item()
                val_list_of_weights.append(weights)

                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
                    
                auc += calculate_auc(outputs.cpu().data.numpy(),
                                               targets.cpu().data.numpy())
                no_iter += 1
                if self.log_freq and (i % self.log_freq == 0):
                    try:
                        self.wandb.log({'Validation loss': val_loss / (i+1), 'Validation AUC': auc / (i+1)})
                    except:
                        pass
                
                group_losses.append(group_loss)
                mean_losses.append(mean_loss)
                losses.append(loss)
                avg_group_losses.append(avg_group_loss)


        auc = 100 * auc / no_iter
        val_loss /= no_iter

        val_dro_results['weights'] = val_list_of_weights
        val_dro_results['group_losses'] = group_losses
        val_dro_results['mean_losses'] = mean_losses
        val_dro_results['losses'] = losses
        val_dro_results['avg_group_losses'] = avg_group_losses

        torch.save(val_dro_results, os.path.join(self.save_path, 'dro_val_loss_epoch_' + str(self.epoch) + '.pth'))

        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
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
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            feature_vectors = []

            for i, (images, targets, sensitive_attr, index) in enumerate(loader):
                images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(
                    self.device)
                outputs, features = self.network.inference(images)
                feature_vectors.append(features.to('cpu'))

                tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
                tol_target += targets.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
                tol_index += index.numpy().tolist()
        
        # save features from test inference
        feature_tensor = torch.cat(feature_vectors)
        torch.save(feature_tensor, os.path.join(self.save_path, 'features.pt'))
        index_tensor = torch.tensor(tol_index)
        torch.save(index_tensor, os.path.join(self.save_path, 'index.pt'))
        print('saved features')

        log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, self.sens_classes)
        overall_FPR, overall_FNR, FPRs, FNRs = calculate_FPR_FNR(pred_df, self.test_meta, self.opt)
        log_dict['Overall FPR'] = overall_FPR
        log_dict['Overall FNR'] = overall_FNR
        pred_df.to_csv(os.path.join(self.save_path, self.experiment + 'pred.csv'), index = False)
        #basics.save_results(t_predictions, tol_target, s_prediction, tol_sensitive, self.save_path)
        for i, FPR in enumerate(FPRs):
            log_dict['FPR-group_' + str(i)] = FPR
        for i, FNR in enumerate(FNRs):
            log_dict['FNR-group_' + str(i)] = FNR
            
        log_dict = basics.add_dict_prefix(log_dict, 'Test ')

        return log_dict