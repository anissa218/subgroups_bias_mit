# Subgroups Matter for Robust Bias Mitigation

This repository contains the code for the ICML 2025 [paper](https://arxiv.org/abs/2505.21363) **"Subgroups Matter for Robust Bias Mitigation."**

NB: Bias mitigation models forked and adapted from [MEDFAIR](https://github.com/ys-zong/MEDFAIR/blob/main/): fairness benchmarking suite for medical imaging. 

## Introduction

In this work, we seek to understand whether subgroup definition may contribute to the recently observed failures of bias mitigation methods. We conduct a comprehensive evaluation of state-of-the-art bias mitigation methods across multiple vision and language classification tasks, systematically varying subgroup definitions, including coarse, fine-grained, intersectional, and noisy subgroups. Our key finding is that **subgroup choice strongly impacts mitigation**, sometimes harming performance more than applying no mitigation at all. Overall, we highlight the importance of careful subgroup definition in bias mitigation and present it as an alternative lever for improving the robustness and fairness of machine learning models.

This repo includes:
- Code to generate biased datasets and different subgroups
- Training code for ERM and bias mitigation baselines
- Tools to measure KL divergence between training/test distributions
- Scripts to reproduce all plots from the paper

## Quick Start

### Installation
Python >= 3.8+ and Pytorch >=1.10 are required for running the code.

```python
cd subgroups_bias_mit/
pip install -r requirements.txt
```

### Dataset

**MNIST** images are freely available open-source and can be downloaded from the following [link](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

**CheXPert** images are also publicly available and can be downloaded through this [website](https://stanfordmlgroup.github.io/competitions/chexpert/). 
Additionally, pacemaker annotations were used, which are kindly provided in this [repository](https://github.com/HarryAnthony/Mahalanobis-OOD-detection).

**CelebA** images can be downloaded from [kaggle](https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset).

**Civil_comments** csv files can also be downloaded from [kaggle](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data).

### Constructing biased datasets and subgroups

To generate the biased training/val datasets and unbiased test dataset and to construct all the subgroup annotations, run the following code. 
```
python preprocessing/make_[mnist/cxp/celeba/civilcomments]_dataset.py --raw_data_folder [path_to_raw_data] --root_folder [root_path] --folder_name [folder_name] --manual_annotations_folder [path_to_manual_annotations (CXP only)]
```

Preprocessed images and splits with the additional metadata are saved in `data/[dataset_name]/pkls` and `data/[dataset_name]/splits` respectively.
 
After preprocessing, specify the paths of the metadata and pickle files in `configs/datasets.json`.

### Run a single experiment

```python
python main.py --experiment [experiment] --wandb_name [experiment_name] --dataset_name [dataset_name] \
     --backbone [backbone] --total_epochs [total_epochs] --sensitive_name [sensitive_name] \
     --batch_size [batch_size] --lr [lr] --sens_classes [sens_classes]  --val_strategy [val_strategy] \
     --output_dim [output_dim] --num_classes [num_classes]
```

See `parse_args.py` for all configurable flags.

### Reproduce our experiments

#### Baseline models

To reproduce all the experiments in the paper, we provide example training scripts for training a baseline model with ERM for the four datasets with our hyperparameters. The arguments `[wandb_name]`, `[data_folder]`, and `[random_seed]` should be changed accordingly.

```python
# MNIST
python main.py --experiment baseline_simple --backbone SimpleCNN --wandb_name [wandb_name] --groupdro_adj 1 --early_stopping 50 --dataset_name MNIST --data_folder [data_folder] --is_small True --total_epochs 50 --batch_size 128 --lr 0.001 --output_dim 1 --num_classes 1

# CXP
python main.py --experiment baseline --early_stopping 10 --backbone cusDenseNet121 --wandb_name [wandb_name] --early_stopping 10 --dataset_name CXP --data_folder [data_folder] --pretrained True --total_epochs 100 --batch_size 256 --lr 0.0005 --output_dim 1 --num_classes 1

# CelebA
python main.py --experiment baseline --backbone cusResNet50 --wandb_name [wandb_name] --early_stopping 5 --dataset_name CELEBA --data_folder [data_folder] --pretrained True --total_epochs 10 --batch_size 256 --lr 0.001 --output_dim 1 --num_classes 1

# Civil_comments
python main.py --experiment baseline --backbone cusBERTClassifier --is_text True --optimizer AdamW --wandb_name [wandb_name] --early_stopping 5 --dataset_name CIVILCOMMENTS --data_folder [data_folder] --pretrained True --total_epochs 10 --batch_size 32 --lr 0.00005 --output_dim 1 --num_classes 1

```

#### Bias mitigation

To train the bias mitigation methods change `--experiment` to `[GroupDRO, resampling, DomainInd, CFair]` and specify the subgroup for mitigation using `[sensitive_name]` and `[sens_classes]` (the number of different subgroups for a given subgrouping). 

Possible subgroups are:
- for gDRO and resampling: `['A','AY','AY_8','S','SY','SY_8','Y','noisy_AY_001','noisy_AY_005','noisy_AY_010','noisy_AY_025','noisy_AY_050','Random','Majority','YAS']`
- for DomainInd: `['A','A_4','S','S_4','AS','Random','Majority','noisy_A_001','noisy_A_005','noisy_A_010','noisy_A_025','noisy_A_050']`
- for CFair: `['A','S','Majority','noisy_A_001','noisy_A_005','noisy_A_010','noisy_A_025','noisy_A_050']`

### Process results

Once all models have trained, process results by running the following commands:

```python
python save_results.py --data ['CheXpert-v1.0-small','mnist','celeba','civilcomments'] --method [mitigation_method] --root_folder [path_to_root_folder] --experiment_folder [parent_dir_where_experiments_are_saved] --data_folder [data_folder] --wandb_name [wandb_name] --random_seed_folders [random_seed_folders]
```
This will save dictionaries containing relevant analyses for each experiment in the `processed_results/` folder.

### Analyse results

We provide example code to analyse results and reproduce the plots made in the paper in the `notebooks/` folder.

### ✅ Reproduction Checklist

- [ ] Download raw datasets
- [ ] Run preprocessing scripts to generate biased datasets and subgroup annotations
- [ ] Set dataset paths in `configs/datasets.json`
- [ ] Train models with `main.py`
- [ ] Run `save_results.py` to process outputs
- [ ] Analyze results using notebooks in `/notebooks`

## Citation
Please consider citing our paper if you find this repo useful.
```
@inproceedings{alloula2025subgroups,
  title     = {Subgroups Matter for Robust Bias Mitigation},
  author    = {Alloula, Anissa and Jones, Charles and Glocker, Ben and Papież, Bartłomiej W.},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.21363},
  note      = {arXiv:2505.21363 [cs.LG]}
}
```

## Acknowledgement

We thank MEDFAIR authors and their detailed repo which provided initial code for this work and Harry Anthony for providing CheXPert pacemaker annotations.
```
@inproceedings{zong2023medfair,
    title={MEDFAIR: Benchmarking Fairness for Medical Imaging},
    author={Yongshuo Zong and Yongxin Yang and Timothy Hospedales},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023},
}
@incollection{Anthony_2023,
	doi = {10.1007/978-3-031-44336-7_14},
	url = {https://doi.org/10.1007%2F978-3-031-44336-7_14},
	year = 2023,
	publisher = {Springer Nature Switzerland},
	pages = {136--146},
	author = {Harry Anthony and Konstantinos Kamnitsas},
	title = {On the Use of Mahalanobis Distance for Out-of-distribution Detection with Neural Networks for Medical Imaging},
	booktitle = {Uncertainty for Safe Utilization of Machine Learning in Medical Imaging}}
```

