import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc, calculate_metrics, get_pred_df
import os
import pandas as pd

from importlib import import_module


def regression_train(opt, network, optimizer, loader, _criterion, wandb):
    """Train the model for one epoch"""
    train_loss, auc, no_iter = 0., 0., 0
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

    for i, (images, targets, sensitive_attr, index) in enumerate(loader):
        images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(opt['device'])
        optimizer.zero_grad()

        outputs, _ = network(images)
        loss = _criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        #auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(), targets.cpu().data.numpy())

        train_loss += loss.item()
        no_iter += 1
        
        if opt['log_freq'] and (i % opt['log_freq'] == 0) and wandb != None:
            wandb.log({'Training loss': train_loss / no_iter}) #, 'Training AUC': auc / no_iter})

        # anissa extra code:
        tol_output += outputs.flatten().cpu().data.numpy().tolist()
        tol_target += targets.cpu().data.numpy().tolist()
        tol_index += index.numpy().tolist()

    #auc = 100 * auc / no_iter
    train_loss /= no_iter

    pred_df = pd.DataFrame(columns=['index', 'pred', 'label','raw_pred'])
    pred_df['index'] = tol_index
    pred_df['pred'] = tol_output
    pred_df['label'] = np.asarray(tol_target).squeeze()

    return auc, train_loss, pred_df


def regression_val(opt, network, loader, _criterion, sens_classes, wandb):
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
            tol_output += outputs.flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
            # auc += calculate_auc(F.sigmoid(outputs).cpu().data.numpy(),
            #                                targets.cpu().data.numpy())
            
            no_iter += 1
            
            if opt['log_freq'] and (i % opt['log_freq'] == 0)  and wandb != None:
                wandb.log({'Validation loss': val_loss / no_iter}) #, 'Validation AUC': auc / no_iter})

    #auc = 100 * auc / no_iter
    #log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    log_dict = {}
    val_loss /= no_iter

    pred_df = pd.DataFrame(columns=['index', 'pred', 'label','raw_pred'])
    pred_df['index'] = tol_index
    pred_df['pred'] = tol_output
    pred_df['label'] = np.asarray(tol_target).squeeze()
    
    return auc, val_loss, log_dict, pred_df


def regression_test(opt, network, loader, _criterion, wandb):
    """Compute model output on testing set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

    with torch.no_grad():
        feature_vectors = []
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images) 
            feature_vectors.append(features.to('cpu'))

            tol_output += outputs.flatten().cpu().data.numpy().tolist()
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

