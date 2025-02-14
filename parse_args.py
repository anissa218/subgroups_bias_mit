from ast import parse
import os
import argparse
from models.basemodels_mlp import cusMLP
import torch
import models
from utils import basics
import wandb
import json
import hashlib
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def collect_args(args=None):
    parser = argparse.ArgumentParser()
    
    # experiments
    parser.add_argument('--experiment',
                        type=str,
                        choices=[
                            'baseline',
                            'CFair',
                            'LAFTR',
                            'LNL',
                            'EnD',
                            'DomainInd',
                            'resampling',
                            'ODR',
                            'SWA',
                            'SWAD',
                            'SAM',
                            'GSAM',
                            'SAMSWAD',
                            'GroupDRO',
                            'BayesCNN',
                            'resamplingSWAD',
                            'Regression',
                            'baseline_simple'
                        ],
                        default = 'baseline')

    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--wandb_name', type=str, default='baseline')
    parser.add_argument('--if_wandb', type=bool, default=True)
    parser.add_argument('--dataset_name', default='CXP', choices=['CXP', 'NIH', 'MIMIC_CXR', 'RadFusion', 'RadFusion4', 
    'HAM10000', 'HAM100004', 'Fitz17k', 'OCT', 'PAPILA', 'ADNI', 'ADNI3T', 'COVID_CT_MD','RadFusion_EHR',
    'MIMIC_III', 'eICU','UKBB_RET','UKBB_RET_REG','MNIST'])

    # class name - just for UKBB
    parser.add_argument('--class_name', default='', choices=['sex', 'bmi', 'ckd', 'bp','adj_bp','diab'])
    parser.add_argument('--data_folder', default = 'all_filt2', help='name of folder from which to get data (eg high quality etc)')

    # is small - just for MNIST
    parser.add_argument('--is_small', type=str2bool, nargs='?',const=False, default=False,help='if using dataset with small image size')   
    parser.add_argument('--resume_path', type = str, default='', help = 'explicitly indentify checkpoint path to resume.')
    
    parser.add_argument('--sensitive_name', default='Sex', choices=['Sex', 'Age', 'Race', 'skin_type', 'Insurance', 'Ethnicity','Centre', 'Random','noisy_001','noisy_005','noisy_010','noisy_025','noisy_050','Age_multi8','Random','Y','Majority','Imperfect','Age_multi4','Imperfect_paper','AY','SY','Artefact','AY_8','SY_8','noisy_AY_001','noisy_AY_005','noisy_AY_010','noisy_AY_025','noisy_AY_050','YAS','noisy_A_001','noisy_A_005','noisy_A_010','noisy_A_025','noisy_A_050','noisy_S_001','noisy_S_005','noisy_S_010','noisy_S_025','noisy_S_050','A_4','S_4','AS'])
    parser.add_argument('--is_3d', type=bool, default=False)
    parser.add_argument('--is_tabular', type=bool, default=False)
    
    parser.add_argument('--adjust_size', type=str2bool, nargs='?',
                        const=False, default=False,
                        help='if using smaller proportion of data') # anissa changes
    parser.add_argument('--dataset_size',type=float,default=100,help='size of dataset to be used relative to full dataset, should be 0.01, 0.02,0.1,0.2,0.5,1.0') # anissa changes
    parser.add_argument('--adjust_centre', type=str2bool, nargs='?',
                        const=False, default=False,
                        help='if excluding one assessment centre') # anissa changes
    parser.add_argument('--centre_name',type=float,default=0,help='centre to exclude from training, should be int from 0-5') # anissa changes

    # training 
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--lr', type=float, default=1e-4, help = 'learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help = 'decay rate of the learning rate')
    parser.add_argument('--lr_decay_period', type=float, default=10, help = 'decay period of the learning rate')
    parser.add_argument('--total_epochs', type=int, default=15, help = 'total training epochs')
    parser.add_argument('--early_stopping', type=int, default=5, help = 'early stopping epochs')
    parser.add_argument('--test_mode', type=bool, default=False, help = 'if using test mode')
    parser.add_argument('--hyper_search', type=bool, default=False, help = 'if searching hyper-parameters')
    parser.add_argument('--augment', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='if using data augmentation') # anissa changes
    parser.add_argument('--pos_weight', type=float, default=1.0, help = 'weight for positive class in loss function')    
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='optimizer for training')
    # testing
    parser.add_argument('--hash_id', type=str, default = '')
    
    # strategy for validation
    parser.add_argument('--val_strategy', type=str, default='loss', choices=['loss', 'worst_auc'], help='strategy for selecting val model')
    parser.add_argument('--gdro_val_loss_for_baseline',type=str, default='None',choices = ['None','Standard','Pure'],help='if using gdro loss for baseline model validation loss calculation')
    
    # cross-domain
    parser.add_argument('--cross_testing', action='store_true')
    parser.add_argument('--source_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T','UKBB_RET','UKBB_RET_REG'])
    parser.add_argument('--target_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T','UKBB_RET','UKBB_RET_REG'])
    parser.add_argument('--cross_testing_model_path', type=str, default='', help='path of the models of three random seeds')
    parser.add_argument('--cross_testing_model_path_single', type=str, default='', help='path of the models')
    
    # network
    parser.add_argument('--backbone', default = 'cusResNet18', choices=['cusResNet18', 'cusResNet50','cusDenseNet121',
                                                                        'cusResNet18_3d', 'cusResNet50_3d', 'cusMLP','SimpleCNN','InceptionV3','ImprovedCNN'])
    #parser.add_argument('--pretrained', type=bool, default=True, help = 'if use pretrained ResNet backbone') 
    parser.add_argument("--pretrained", type=str2bool, nargs='?',
                        const=True, default=True,
                        help='if use pretrained ResNet backbone') # anissa changes

    parser.add_argument('--output_dim', type=int, default=14, help='output dimension of the classification network')
    parser.add_argument('--num_classes', type=int, default=14, help='number of target classes')
    parser.add_argument('--sens_classes', type=int, default=2, help='number of sensitive classes')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel of the images')
    
    # resampling
    parser.add_argument('--resample_which', type=str, default='group', choices=['class', 'balanced'], help='audit step for LAFTR')
    
    # LAFTR
    parser.add_argument('--aud_steps', type=int, default=1, help='audit step for LAFTR')
    parser.add_argument('--class_coeff', type=float, default=1.0, help='coefficient for classification loss of LAFTR')
    parser.add_argument('--fair_coeff', type=float, default=1.0, help='coefficient for fair loss of LAFTR')
    parser.add_argument('--model_var', type=str, default='laftr-eqodd', help='model variation for LAFTR')
    # CFair
    
    # LNL
    parser.add_argument('--_lambda', type=float, default=0.1, help='coefficient for loss of LNL')

    # EnD
    parser.add_argument('--alpha', type=float, default=0.1, help='weighting parameters alpha for EnD method')
    parser.add_argument('--beta', type=float, default=0.1, help='weighting parameters beta for EnD method')

    # ODR
    parser.add_argument("--lambda_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--lambda_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--step_size", type=int, default=20, help="step size for adjusting coefficients for loss of ODR")
    
    # GroupDRO
    parser.add_argument("--groupdro_alpha", type=float, default=1.0, help="coefficient alpha for loss of GroupDRO if doing robust loss greedy (if =1 no robust loss greedy)")
    parser.add_argument("--groupdro_gamma", type=float, default=0.1, help="coefficient gamma for loss of GroupDRO")
    parser.add_argument("--groupdro_step", type=float, default=0.01, help="step size GroupDRO (how much to weight high losses)")
    parser.add_argument("--groupdro_adj", type=float, default=1, help="how much to adjust for group size in GroupDRO")
    parser.add_argument("--groupdro_pure", type=str2bool, nargs='?',
                        const=False, default=False, help="if doing pure GroupDRO")
    parser.add_argument("--groupdro_use_train_loss_for_val", type=str2bool, nargs='?',
                        const=False, default=False, help="if using train gDRO loss instantiation for val too")
    parser.add_argument("--groupdro_pure_val_loss", type=str2bool, nargs='?',
                    const=False, default=False, help="if doing pure GroupDRO for val loss")

    # SWA
    parser.add_argument("--swa_start", type=int, default=7, help="starting epoch for averaging of SWA")
    parser.add_argument("--swa_lr", type=float, default=0.0001, help="learning rate for averaging of SWA")
    parser.add_argument("--swa_annealing_epochs", type=int, default=3, help="learning rate for averaging of SWA")
    # SWAD
    parser.add_argument("--swad_n_converge", type=int, default=3, help="starting converging epoch of SWAD")
    parser.add_argument("--swad_n_tolerance", type=int, default=6, help="tolerance steps of SWAD")
    parser.add_argument("--swad_tolerance_ratio", type=float, default=0.05, help="tolerance ratio of SWAD")
    
    # SAM
    parser.add_argument("--rho", type=float, default=2, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", type=bool, default=True, help="whether using adaptive mode for SAM.")
    parser.add_argument("--T_max", type=int, default=50, help="Value for LR scheduler")
    
    # GSAM
    parser.add_argument("--gsam_alpha", type=float, default=2, help="Rho parameter for SAM.")

    # BayesCNN
    parser.add_argument("--num_monte_carlo", type=int, default=10, help="Rho parameter for SAM.")
    
    parser.set_defaults(cuda=True)
    
    # logging 
    parser.add_argument('--log_freq', type=int, default=50, help = 'logging frequency (step)')
    
    # anissa test (to run in jupyter notebook)
    if args is None:
        opt = vars(parser.parse_args())
    else:
        opt = vars(parser.parse_args(args))

    opt = create_exerpiment_setting(opt)
    return opt


def create_exerpiment_setting(opt, do_wandb=True):
    
    # get hash
    
    run_hash = hashlib.sha1()
    run_hash.update(str(time.time()).encode('utf-8'))
    opt['hash'] = run_hash.hexdigest()[:10]
    print('run hash (first 10 digits): ', opt['hash'])
    
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    
    opt['save_folder'] = os.path.join('your_path/fariness_data/model_records', opt['dataset_name'], opt['sensitive_name'], opt['backbone'], opt['experiment'],opt['wandb_name'],str(opt['random_seed']))
    opt['resume_path'] = opt['save_folder']
    basics.creat_folder(opt['save_folder'])
    
    optimizer_setting = {
        'optimizer': torch.optim.SGD if opt['optimizer'] == 'SGD' else torch.optim.Adam,
        'lr': opt['lr'],
        'weight_decay': opt['weight_decay'],
    }
    opt['optimizer_setting'] = optimizer_setting

    print('early stopping: ', opt['early_stopping'])
    
    optimizer_setting2 = {
        'optimizer': torch.optim.Adam,
        'lr': opt['lr'],
        'weight_decay': opt['weight_decay'],
    }
    opt['optimizer_setting2'] = optimizer_setting2
    
    opt['dropout'] = 0.5

    # dataset configurations
    if opt['cross_testing']:
        opt['dataset_name'] = opt['target_domain']
    
    with open('configs/datasets.json', 'r') as f:
        data_path = json.load(f)

    try:
        data_setting = data_path[opt['dataset_name']]

        if opt['augment'] == False:
            data_setting['augment'] = False
        else:
            data_setting['augment'] = True

        if opt['adjust_size'] == True:
            data_setting['adjust_size'] = True
            data_setting['dataset_size'] = float(opt['dataset_size'])
        
        else:
            data_setting['adjust_size'] = False
        if opt['is_small'] == True:
            data_setting['is_small'] = True
        else:
            data_setting['is_small'] = False
        try:
            if opt['adjust_centre'] == True:
                data_setting['adjust_centre'] = True
                data_setting['centre_name'] = int(opt['centre_name'])
            else:
                data_setting['adjust_centre'] = False
        except:
            pass

    except:
        data_setting = {}
    
    opt['data_setting'] = data_setting
    
    # experiment-specific setting
    
    if opt['experiment'] == 'DomainInd':
        opt['output_dim'] *= opt['sens_classes']
    if opt['experiment'] == 'LAFTR' or opt['experiment'] == 'CFair':
        opt['train_sens_classes'] = 2
    else:
        opt['train_sens_classes'] = opt['sens_classes']

    if do_wandb == False:
        wandb=None
        return opt,wandb
    else:
        import wandb
        if opt['if_wandb'] == True:
            with open('configs/wandb_init.json') as f:
                wandb_args = json.load(f)
            wandb_args["tags"] = [opt['hash']]
            wandb_args["name"] = opt['wandb_name'] # anissa changed this
            wandb.init(**wandb_args, config = opt)
        else:
            wandb = None
        
    return opt, wandb
