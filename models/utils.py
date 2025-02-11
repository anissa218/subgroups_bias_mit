import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc, calculate_metrics, get_pred_df
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from importlib import import_module

def standard_train(opt, network, optimizer, loader, _criterion, wandb,scheduler=None):
    """Train the model for one epoch"""
    train_loss, auc, no_iter = 0., 0., 0
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    print('start standard train')

    for i, (images, targets, sensitive_attr, index) in enumerate(loader):
        images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(opt['device'])
        optimizer.zero_grad()
        outputs, _ = network(images)
        loss = _criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())

        train_loss += loss.item()
        no_iter += 1
        
        if opt['log_freq'] and (i % opt['log_freq'] == 0) and wandb != None:
            wandb.log({'Training loss': train_loss / no_iter, 'Training AUC': auc / no_iter})

        tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
        tol_target += targets.cpu().data.numpy().tolist()
        tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
        tol_index += index.numpy().tolist()

    if scheduler is not None:
        print('scheduler step')
        scheduler.step()

    auc = 100 * auc / no_iter
    train_loss /= no_iter

    pred_df = get_pred_df(tol_output, tol_target, tol_sensitive, tol_index)

    return auc, train_loss, pred_df


def standard_val(opt, network, loader, _criterion, sens_classes, wandb):
    """Compute model output on validation set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
    val_loss, auc = 0., 0.
    no_iter = 0
    with torch.no_grad():

        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)

            loss = _criterion(outputs, targets)
            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()
            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                           targets.cpu().data.numpy())
            
            no_iter += 1
            
            if opt['log_freq'] and (i % opt['log_freq'] == 0)  and wandb != None:
                wandb.log({'Validation loss': val_loss / no_iter, 'Validation AUC': auc / no_iter})


    auc = 100 * auc / no_iter
    val_loss /= no_iter
    log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    
    return auc, val_loss, log_dict, pred_df


def standard_test(opt, network, loader, _criterion, wandb):
    """Compute model output on testing set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

    with torch.no_grad():
        feature_vectors = []
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.inference(images) 

            feature_vectors.append(features.to('cpu'))

            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
    
        # save features from test inference
        feature_tensor = torch.cat(feature_vectors)
        torch.save(feature_tensor, os.path.join(opt['save_folder'], 'features.pt'))
        index_tensor = torch.tensor(tol_index)
        torch.save(index_tensor, os.path.join(opt['save_folder'], 'index.pt'))
        print('saved features')


    return tol_output, tol_target, tol_sensitive, tol_index

def gdro_val(opt, network, loader, _criterion, sens_classes, wandb,val_data,pure_loss=False):
    from models.GroupDRO.utils import LossComputer # is this v bad practice?
    """Compute model output on validation set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    
    val_loss, auc = 0., 0.
    no_iter = 0
    val_loss_computer = LossComputer(
            criterion = _criterion,
            is_robust=True,
            dataset=val_data,
            alpha=1,
            gamma=0.1,
            step_size=0.01,
            normalize_loss=False,
            btl=False, # have set it to always false (same as original implementation)
            min_var_weight=0,
            pure = pure_loss) 
    
    with torch.no_grad():

        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)
            loss = _criterion(outputs, targets)
            group_loss,mean_loss,loss,worst_index,weights = val_loss_computer.loss(outputs, targets, sensitive_attr, is_training=False)
            print('group_loss',group_loss)
            print('loss',loss)
            print('worst_index',worst_index)
            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()

            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
            auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
                                           targets.cpu().data.numpy())
            
            no_iter += 1
            
            if opt['log_freq'] and (i % opt['log_freq'] == 0)  and wandb != None:
                wandb.log({'Validation loss': val_loss / no_iter, 'Validation AUC': auc / no_iter})


    auc = 100 * auc / no_iter
    val_loss /= no_iter
    log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    
    return auc, val_loss, log_dict, pred_df

