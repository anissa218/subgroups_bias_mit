import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.manifold import TSNE
from utils.results_utils import *

import torch

def get_mnist_test_results(root='/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR',experiment_folder = 'your_path/fariness_data/model_records/MNIST/',model_name = 'SimpleCNN/GroupDRO/o_ay_sens_classes_s1e-2_a1', data_folder = 'two_sens_classes',random_seed_folders = ['42','43','44'],experiments=None,pred_file = 'GroupDROpred.csv',subgroups=['Age_binary'],use_val_threshold=True,val_threshold_dict={},preprocessing_function=preprocess_mnist_data,data='mnist'):
    '''
    Get all results from a given experiment folder

    Parameters:
    root: str
        root path of the project
    experiment_folder: str
        path to the experiment folder
    model_name: str
        path with name of model and params etc
    data_folder: str
        name of the data folder
    random_seed_folders: list
        list of random seed folders, can be multiple or just one
    experiments: list
        list of experiments (only specify if you don't just want all the expeirments in your experiment_folder)
    pred_file: str
        name of the file where preds are stored
    subgroups: list
        subgroups wrt to calculate disparities
    
    Returns:
    test_results_dict: dict
        dictionary containing all the results on test data

    '''

    if experiments == None:
        experiments = os.listdir(os.path.join(root,experiment_folder))
    else:
        experiments = experiments

    path_to_test_splits = os.path.join(root,'data',data,data_folder,'splits','test.csv')

    test_results_dict = {}

    
    for experiment in experiments:
        print(experiment)
        for random_seed in random_seed_folders:
            results_folder = os.path.join(root,experiment_folder,experiment,model_name,random_seed)

            if use_val_threshold:
                test_preds = preprocessing_function(os.path.join(results_folder,pred_file),path_to_test_splits,threshold_function=None,threshold = val_threshold_dict[experiment][random_seed])

            else:
                test_preds = preprocessing_function(os.path.join(results_folder,pred_file),path_to_test_splits)

            if 'AY' not in test_preds.columns:
                test_preds['AY'] = test_preds['binaryLabel'].astype(str) + test_preds['Age_binary'].astype(str)

            test_results_df = make_test_results_df([test_preds],subgroups,calculate_auc=True)
            test_results_df.set_index('Subgroup',inplace=True)

            overall_test_results_df = pd.DataFrame(test_results_df.iloc[0][['Test Acc','Test AUC','Test Precision','Test Recall']])
            overall_test_results_df.rename(columns={'Age_binary':experiment},inplace=True)

            if experiment not in test_results_dict:
                test_results_dict[experiment] = {}

            test_results_dict[experiment][random_seed] = {
                'test_preds': test_preds,
                'overall_test_results': overall_test_results_df
            }     
                   
    return test_results_dict


def get_mnist_train_val_results(root='/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR',experiment_folder = 'your_path/fariness_data/model_records/MNIST/',model_name = 'SimpleCNN/GroupDRO/o_ay_sens_classes_s1e-2_a1', data_folder = 'two_sens_classes',random_seed_folders = ['42','43','44'],experiments=None,pred_file = 'GroupDROpred.csv',subgroups=['Age_binary'],threshold_function = find_balanced_acc_threshold,preprocessing_function=preprocess_mnist_data,data='mnist'):
    '''
    Get all train val and loss results from a given experiment folder

    Parameters:
    same as get_mnist_test_results
    
    Returns:
    train_val_results_dict: dict
        dictionary containing all the results on train and val data

    '''
    if experiments == None:
        experiments = os.listdir(os.path.join(root,experiment_folder))
    else:
        experiments = experiments

    path_to_train_splits = os.path.join(root,'data',data,data_folder,'splits','train.csv')
    path_to_val_splits = os.path.join(root,'data',data,data_folder,'splits','val.csv')
    path_to_test_splits = os.path.join(root,'data',data,data_folder,'splits','test.csv')

    train_val_dict = {}

    for experiment in experiments:
        print(experiment)
        for random_seed in random_seed_folders:
            
            results_folder = os.path.join(root,experiment_folder,experiment,model_name,random_seed)

            train_preds,val_preds = get_val_train_preds(results_folder,path_to_train_splits,path_to_val_splits,preprocessing_function=preprocessing_function,threshold_function = threshold_function)

            for key in train_preds.keys():
                if 'AY' not in train_preds[key].columns:
                    train_preds[key]['AY'] = train_preds[key]['binaryLabel'].astype(str) + train_preds[key]['Age_binary'].astype(str)
                    val_preds[key]['AY'] = val_preds[key]['binaryLabel'].astype(str) + val_preds[key]['Age_binary'].astype(str)
                
            if experiment not in train_val_dict:
                train_val_dict[experiment] = {}

            train_val_dict[experiment][random_seed] = {
                'train_preds': train_preds,
                'val_preds': val_preds
            }     
             
    return train_val_dict

def get_mnist_loss_results(root='/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR',experiment_folder = 'your_path/fariness_data/model_records/MNIST/',model_name = 'SimpleCNN/GroupDRO/o_ay_sens_classes_s1e-2_a1', data_folder = 'two_sens_classes',random_seed_folders = ['42','43','44'],experiments=None,pred_file = 'GroupDROpred.csv',subgroups=['Age_binary'],is_val_loss=False):
    '''
    Get loss results from a given experiment folder

    Parameters:
    same as get_mnist_test_results
    
    Returns:
    loss_dict: dict
        dictionary containing all the results on train and val data

    '''
    if experiments == None:
        experiments = os.listdir(os.path.join(root,experiment_folder))
    else:
        experiments = experiments

    loss_dict = {}

    for experiment in experiments:
        print(experiment)
        for random_seed in random_seed_folders:
            results_folder = os.path.join(root,experiment_folder,experiment,model_name,random_seed)
            n_epochs = len([x for x in os.listdir(os.path.join(results_folder)) if 'dro_loss_epoch' in x])

            if is_val_loss:
                epoch_losses = [f'dro_val_loss_epoch_{x+1}.pth' for x in range(n_epochs)]
            else:
                epoch_losses = [f'dro_loss_epoch_{x}.pth' for x in range(n_epochs)]

            group_losses_list = []
            mean_losses_list = []
            losses_list = []
            avg_group_losses_list = []
            group_losses = []
            mean_losses = []
            losses = []
            avg_group_losses = []
            weights = []
            weights_list = []

            for epoch_loss in epoch_losses:
                loaded_dict = torch.load(os.path.join(results_folder,epoch_loss),map_location=torch.device('cpu'))

                group_losses_list = [tensor.detach().numpy() for tensor in loaded_dict['group_losses']]
                mean_losses_list = [tensor.detach().numpy() for tensor in loaded_dict['mean_losses']]
                losses_list = [tensor.detach().numpy() for tensor in loaded_dict['losses']]
                avg_group_losses_list = [tensor.detach().numpy() for tensor in loaded_dict['avg_group_losses']]

                if 'weights' in loaded_dict.keys():
                    weights_list = [tensor.detach().numpy() for tensor in loaded_dict['weights']]
                    weights.append(np.mean(weights_list,axis=0))

                
                # take mean over all the batches of that epoch
                mean_group_losses = np.mean(group_losses_list,axis=0)
                mean_mean_losses = np.mean(mean_losses_list,axis=0)
                mean_dro_losses = np.mean(losses_list,axis=0)
                mean_avg_group_losses = np.mean(avg_group_losses_list,axis=0)

                group_losses.append(mean_group_losses)
                mean_losses.append(mean_mean_losses)
                losses.append(mean_dro_losses)
                avg_group_losses.append(mean_avg_group_losses)
                            
            if experiment not in loss_dict:
                loss_dict[experiment] = {}

            loss_dict[experiment][random_seed] = {
                'group_losses': group_losses,
                'mean_losses': mean_losses,
                'losses': losses,
                'avg_group_losses': avg_group_losses,
                'weights': weights
            } 
    return loss_dict         