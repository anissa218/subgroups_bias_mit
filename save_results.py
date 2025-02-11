import os
import pandas as pd
import numpy as np
from utils.results_utils import *
from utils.save_results_utils import *
import torch
import torch.nn as nn
import seaborn as sns
import pickle
from multiprocessing import Pool
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process and save CXP and MNIST results.")
    parser.add_argument("--data", type=str, default = 'CheXpert-v1.0-small', choices = ['CheXpert-v1.0-small','mnist'], help="Data to use: CheXpert-v1.0-small or mnist")
    parser.add_argument('--method', type=str, required=True, choices = ['GroupDRO', 'baseline', 'resampling', 'DomainInd', 'CFair'], help="Method to use: GroupDRO, DomainInd, CFair, resampling or baseline.")
    parser.add_argument("--root_folder", type=str, default = '/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR', help="Path to the root folder (where data, results etc. are all saved) ")
    parser.add_argument("--experiment_folder", type=str, default = 'your_path/fariness_data/model_records/', help="Path to parent directory where model runs are stored")
    parser.add_argument("--data_folder", type=str, default = 'engineered_bias_pacemaker_is_1', help="Data configuration used for experiments") # or 'two_variables
    parser.add_argument("--random_seed_folders", nargs = '+', default = ['42','43','44'], help="Random seeds used for experiments")
    parser.add_argument("--wandb_name", type=str, default = 'ebp1_bs256_lr5e4_features', help="Name to identify run and processed results")
    parser.add_argument("--experiments", type=str, default = [],help = "list of subgroup experiments to analyse")
    parser.add_argument("--model_backbone", type=str, default = [], help="Model backbone used for experiments")
    return parser.parse_args()

if __name__ == "__main__":

    # VARIABLES

    args = parse_args()

    data = args.data
    method = args.method
    root_folder = args.root_folder
    experiment_folder = args.experiment_folder
    data_folder = args.data_folder
    random_seed_folders = args.random_seed_folders
    wandb_name = args.wandb_name
    experiments = args.experiments
    model_backbone = args.model_backbone

    if data == 'CheXpert-v1.0-small':
        data_type = 'CXP'
        preprocessing_function = preprocess_chexpert_data
        if model_backbone == []:
            model_backbone = 'cusDenseNet121'
    elif data == 'mnist':
        data_type = 'MNIST'
        preprocessing_function = preprocess_mnist_data
        if model_backbone == []:
            model_backbone = 'SimpleCNN'

    experiment_folder = os.path.join(experiment_folder, data_type) #Age/SimpleCNN/GroupDRO'

    if experiments == []: # default experiments
        experiments = ['Artefact','AY','AY_8','Sex','SY','SY_8','Y','noisy_AY_001','noisy_AY_005','noisy_AY_010','noisy_AY_025','noisy_AY_050','Random','Majority','YAS']
        #experiments = ['AY','SY']
        if method == 'CFair':
            experiments = ['Artefact','Sex','Majority','noisy_A_001','noisy_A_005','noisy_A_010','noisy_A_025','noisy_A_050']
        if method == 'DomainInd':
            experiments = ['Artefact','A_4','Sex','S_4','AS','Random','Majority','noisy_A_001','noisy_A_005','noisy_A_010','noisy_A_025','noisy_A_050']

    subgroups = ['Sex_binary']

    model_name = os.path.join(model_backbone,method,wandb_name)

    if method == 'baseline' or method == 'resampling' or method == 'EnD' or method == 'CFair':
        pred_file = 'pretrained_pred.csv'
    elif method == 'ODR':
        pred_file = 'pred.csv'
    elif method == 'GroupDRO':
        pred_file = 'GroupDROpred.csv'
    elif method == 'DomainInd':
        pred_file = 'DomainInd_pred.csv'
    else:
        print('Method not recognized')
        exit()
    
    # ANALYSE INF RESULTS ON TEST DATA

    val_threshold_dict = {}

    for experiment in experiments:
        val_threshold_dict[experiment] = {}
        for random_seed in random_seed_folders:
            results_folder = os.path.join(root_folder,experiment_folder,experiment,model_name,random_seed)

            val_df = pd.read_csv(os.path.join(results_folder,'best_val_pred.csv'))
            val_threshold = find_balanced_acc_threshold(val_df['raw_pred'],val_df['label'])

            val_threshold_dict[experiment][random_seed] = val_threshold
        
    test_results_dict = get_mnist_test_results(root_folder, experiment_folder, model_name, data_folder,random_seed_folders,experiments,pred_file,subgroups,True,val_threshold_dict,preprocessing_function=preprocessing_function,data=data)

    # ANALYSE TRAIN AND VAL RESULTS

    if method == 'baseline' or method == 'GroupDRO':
        train_val_dict = get_mnist_train_val_results(root_folder, experiment_folder, model_name, data_folder,random_seed_folders,experiments,pred_file,subgroups,threshold_function=find_balanced_acc_threshold,preprocessing_function=preprocessing_function,data=data)

    # ANALYSE LOSS RESULTS (ONLY FOR GROUPDRO)

    if method == 'GroupDRO':
        loss_dict = get_mnist_loss_results(root_folder, experiment_folder, model_name, data_folder,random_seed_folders,experiments,pred_file,subgroups)
        with open(f'{data}_{method}_{wandb_name}_loss_dict_test.pkl', 'wb') as f:
            pickle.dump(loss_dict, f)
        
        # also add results from baseline model if 'groupDRO'

        experiments = ['AY']
        experiment = experiments[0]
        pred_file = 'pretrained_pred.csv'
        model_name = os.path.join(model_backbone, 'baseline', wandb_name)

        baseline_train_val_results_dict = get_mnist_train_val_results(root_folder, experiment_folder, model_name, data_folder,random_seed_folders,experiments,pred_file,subgroups,preprocessing_function=preprocessing_function,data=data)
        
        val_threshold_dict[experiment] = {}
        for random_seed in random_seed_folders:
            results_folder = os.path.join(root_folder,experiment_folder,experiment,model_name,random_seed)

            val_df = pd.read_csv(os.path.join(results_folder,'best_val_pred.csv'))
            val_threshold = find_balanced_acc_threshold(val_df['raw_pred'],val_df['label'])

            val_threshold_dict[experiment][random_seed] = val_threshold
        
        baseline_test_results_dict = get_mnist_test_results(root_folder, experiment_folder, model_name, data_folder,random_seed_folders,experiments,pred_file,subgroups,True,val_threshold_dict,preprocessing_function=preprocessing_function,data=data)

        baseline_test_results_dict['baseline'] = baseline_test_results_dict.pop(experiments[0])
        baseline_train_val_results_dict['baseline'] = baseline_train_val_results_dict.pop(experiments[0])

        test_results_dict.update(baseline_test_results_dict)
        train_val_dict.update(baseline_train_val_results_dict)

        with open(f'processed_results/{data}_{method}_{wandb_name}_train_val_results_dict_test.pkl', 'wb') as f:
            pickle.dump(train_val_dict, f)


    with open(f'processed_results/{data}_{method}_{wandb_name}_test_results_dict_test.pkl', 'wb') as f:
        pickle.dump(test_results_dict, f)
