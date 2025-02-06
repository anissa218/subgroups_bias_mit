# Subgroups Matter for Robust Bias Mitigation

Code for paper submitted to ICML 2025.

Bias mitigation models forked and adapted from [MEDFAIR](https://github.com/ys-zong/MEDFAIR/blob/main/): fairness benchmarking suite for medical imaging ([paper](https://arxiv.org/abs/2210.01725)). 

## Quick Start

### Installation
Python >= 3.8+ and Pytorch >=1.10 are required for running the code.

```python
cd MEDFAIR/
pip install -r my_requirements.txt
```

### Dataset

**MNIST** images are freely available open-source and can be downloaded from the following [link](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

**CheXPert** images are also publicly available and can be downloaded through this [website](https://stanfordmlgroup.github.io/competitions/chexpert/). 
Additionally, pacemaker annotations were used, which are kindly provided in this [repository](https://github.com/HarryAnthony/Mahalanobis-OOD-detection).

### Constructing biased datasets and subgroups

To generate the biased training/val datasets and unbiased test dataset and to construct all the subgroup annotations, run the follwoing code. 
```
python make_mnist_dataset.py --raw_data_folder [path_to_raw_data] --root_folder [root_path] --folder_name [folder_name]
python make_cxp_dataset.py --raw_data_folder [path_to_raw_data] --manual_annotations_folder [path_to_manual_annotations] --root_folder [root_path] --folder_name [folder_name]
```

Preprocessed images and splits with the additional metadata are saved in data/[dataset_name]/pkls and data/[dataset_name]/respectively
 
After preprocessing, specify the paths of the metadata and pickle files in `configs/datasets.json`.

### Run a single experiment

```python
python main.py --experiment [experiment] --experiment_name [experiment_name] --dataset_name [dataset_name] \
     --backbone [backbone] --total_epochs [total_epochs] --sensitive_name [sensitive_name] \
     --batch_size [batch_size] --lr [lr] --sens_classes [sens_classes]  --val_strategy [val_strategy] \
     --output_dim [output_dim] --num_classes [num_classes]
```

See `parse_args.py` for more options.

### Reproduce our experiments

To reproduce all the MNIST and CXP experiments in the paper, run the following code for mitigation experiments in [GroupDRO, resampling, DomainInd, CFair] and varying the subgroup for mitigation and sens_classes accordingly. Also change [wandb_name], [data_folder], and [random_seed] accordingly.


```python
## MNIST ##
# baseline model
python main.py --experiment baseline_simple --backbone SimpleCNN --wandb_name [wandb_name] --groupdro_adj 1 --early_stopping 50 --dataset_name MNIST --data_folder [data_folder] --is_small True --total_epochs 50 --batch_size 128 --lr 0.001 --output_dim 1 --num_classes 1  --random_seed [random_seed]
# mitigation model
python main.py --experiment [mitigation_method] --backbone SimpleCNN --wandb_name [wandb_name] --early_stopping 50 --dataset_name MNIST --data_folder [data_folder] --is_small True --total_epochs 50 --sensitive_name [subgroup] --sens_classes [n_subgroups] --batch_size 128 --lr 0.001 --output_dim 1 --num_classes 1  --random_seed [random_seed]

## CXP ##
# baseline model
python main.py --experiment baseline --early_stopping 10 --backbone cusDenseNet121 --wandb_name [wandb_name] --early_stopping 10 --dataset_name CXP --data_folder [data_folder] --pretrained True --total_epochs 100 --batch_size 256 --lr 0.0005 --output_dim 1 --num_classes 1 --random_seed [random_seed]
# mitigation model
python main.py --experiment [mitigation_method] --early_stopping 10 --backbone cusDenseNet121 --wandb_name [wandb_name] --early_stopping 10 --dataset_name CXP --data_folder [data_folder] --pretrained True --total_epochs 100 --sensitive_name [subgroup] --sens_classes [n_subgroups] --batch_size 256 --lr 0.0005 --output_dim 1 --num_classes 1 --random_seed [random_seed]
```


### Process results

Once all models have trained, 

## Citation
Please consider citing our paper if you find this repo useful.
```
TODO
```
## Acknowledgement

We thank MEDFAIR authors and their detailed repo which provided initial code for this work.
```
@inproceedings{zong2023medfair,
    title={MEDFAIR: Benchmarking Fairness for Medical Imaging},
    author={Yongshuo Zong and Yongxin Yang and Timothy Hospedales},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023},
}
```

