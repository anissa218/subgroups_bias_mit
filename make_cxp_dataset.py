import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
from sklearn.model_selection import train_test_split
import torch
import argparse
import cv2
import pickle
from pathlib import Path

from utils.make_datasets import *

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CXP dataset with bias.")
    parser.add_argument("--raw_data_folder", type=str, required=True, help="Path to the raw CXP data folder (assumes this contains a 'train.csv' file) ")
    parser.add_argument("--path_to_annotations", type=str, required=True, help="Path to the annotations folder")
    parser.add_argument("--save_data_folder", type=str, required=True, help="Root directory where processed data will be saved")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder name for saving the dataset")
    return parser.parse_args()

if __name__ == "__main__":

    # variables
    
    args = parse_args()

    raw_data_folder = args.raw_data_folder
    path_to_annotations = args.path_to_annotations
    save_data_folder = args.save_data_folder
    folder_name = args.folder_name

    PACEMAKER_IS_1 = True

    # params

    size_per_group_test = 70
    prop_val_of_train_val = 0.125

    prop_positive_images_train_val = 0.5
    prop_male_images_train_val = 0.5
    prop_spurious_images_male_train_val = 0.95
    prop_spurious_images_female_train_val = 0.8

    prop_000 = prop_spurious_images_male_train_val*1/4
    prop_001 = prop_spurious_images_female_train_val*1/4
    prop_010 = 1/4 - prop_000
    prop_011 = 1/4 - prop_001

    prop_110 = prop_spurious_images_male_train_val*1/4
    prop_111 = prop_spurious_images_female_train_val*1/4
    prop_100 = 1/4 - prop_110
    prop_101 = 1/4 - prop_111

    AY_map = {
        (0, 0): 0,  # Artefact=0, binaryLabel=0 -> AY=0
        (0, 1): 1,  # Artefact=0, binaryLabel=1 -> AY=1
        (1, 0): 2   # Artefact=1, binaryLabel=0 -> AY=2
    }
    default_AY = 3  # Default value for AY

    SY_map = {
        ('M', 0): 0,  # Sex='M', binaryLabel=0 -> SY=0
        ('M', 1): 1,  # Sex='M', binaryLabel=1 -> SY=1
        ('F', 0): 2   # Sex='F', binaryLabel=0 -> SY=2
    }

    AS_map = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    YAS_map = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 0): 2,
        (0, 1, 1): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7,
    }
    default_SY = 3  # Default value for SY(subset_train_images_df['Sex'] == 'F') & (subset_train_images_df['binaryLabel'] == 0)
    AS_map = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }

    choices = [0, 1, 2]

    # Load data

    np.random.seed(42)
    torch.manual_seed(42)

    train_metadata_path = os.path.join(raw_data_folder, 'train.csv')
    metadata_df = pd.read_csv(train_metadata_path)
    metadata_df['subject_id'] = metadata_df['Path'].apply(lambda x: x.split('/')[2])
    metadata_df['study_id'] = metadata_df['Path'].apply(lambda x: x.split('/')[3])
    metadata_df['Path'] = metadata_df['Path'].apply(lambda x: os.path.join(Path(raw_data_folder).parent,x) )

    pacemaker_list = np.loadtxt(os.path.join(path_to_annotations,'pacemaker.txt'),dtype=str)
    pacemaker_list = [os.path.join(raw_data_folder,str(element)) for element in pacemaker_list ] # add the full path

    no_support_device_list = np.loadtxt(os.path.join(path_to_annotations,'no_support_device.txt'),dtype=str)
    no_support_device_list = [os.path.join(raw_data_folder,str(element)) for element in no_support_device_list ]

    annotations_list = np.concatenate((pacemaker_list,no_support_device_list))

    subset_df = metadata_df.copy()
    subset_df = subset_df[subset_df['Path'].isin(annotations_list)]

    if PACEMAKER_IS_1:
        subset_df['Pacemaker'] = subset_df['Path'].apply(lambda x: 1 if x in pacemaker_list else 0)
    else:
        subset_df['Pacemaker'] = subset_df['Path'].apply(lambda x: 0 if x in pacemaker_list else 1)

    subset_df['binary_label'] = subset_df['Pleural Effusion'].apply(lambda x: 1 if x == 1 else 0)
    subset_df['Sex_binary'] = subset_df['Sex'].apply(lambda x: 1 if x == 'Female' else 0)

    # add some more groups

    np.random.seed(42)

    subset_df['Y'] = subset_df['binary_label']
    subset_df['Artefact'] = subset_df['Pacemaker']
    subset_df['AY'] = subset_df.apply(
        lambda row: AY_map.get((row['Artefact'], row['binary_label']), default_AY), axis=1)
    subset_df['SY'] = subset_df.apply(
        lambda row: AY_map.get((row['Sex_binary'], row['binary_label']), default_AY), axis=1) # same map as AY
    subset_df['AY_8'] = [x if random.random() < 0.5 else x + 4 for x in subset_df['AY']]
    subset_df['SY_8'] = [x if random.random() < 0.5 else x + 4 for x in subset_df['SY']]
    subset_df['Random'] = np.random.choice([0, 1, 2, 3], size=len(subset_df))
    subset_df['Majority'] = [0 if x == 0 or x == 3 else 1 for x in subset_df['AY']]
    subset_df['YAS'] = subset_df[['Y','Pacemaker','Sex_binary']].apply(tuple, axis=1).map(YAS_map)
    subset_df['Sex'] = subset_df['Sex'].apply(lambda x: 'F' if x == 'Female' else 'M')

    # Adding noisy columns

    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_AY_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise_with_proportions(subset_df, 'AY', error_percent)['AY']

    subset_df['AS'] = subset_df[['Pacemaker','Sex_binary']].apply(tuple, axis=1).map(AS_map)

    subset_df['A_4'] = [x if random.random() < 0.5 else x + 2 for x in subset_df['Pacemaker']]
    subset_df['S_4'] = [x if random.random() < 0.5 else x + 2 for x in subset_df['Sex_binary']]

    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_A_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise(subset_df, 'Pacemaker', error_percent)['Pacemaker']
        
    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_S_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise(subset_df, 'Sex_binary', error_percent)['Sex_binary']
        
    test_set_df = subset_df.groupby('YAS', group_keys=False).apply(lambda x: x.sample(n=size_per_group_test, random_state=42))

    remaining_df = subset_df.drop(test_set_df.index)

    total_images = 2665.0 # this amount allows you to respect the proportions while not exceeding the max number for each Y,A,S group

    samples_per_group = { 
        0: int(total_images * prop_000),
        1: int(total_images * prop_001),
        2: int(total_images * prop_010),
        3: int(total_images * prop_011),
        4: int(total_images * prop_110),
        5: int(total_images * prop_111),  # All available images
        6: int(total_images * prop_100),
        7: int(total_images * prop_101),
    } # Calculate the number of images to sample for each group

    # Ensure no group is oversampled
    # for group in samples_per_group:
    #     available_images = len(remaining_df[remaining_df['YAS'] == group])
    #     samples_per_group[group] = min(samples_per_group[group], available_images)

    train_val_df = pd.concat(
        [
            remaining_df[remaining_df['YAS'] == group].sample(
                n=samples_per_group[group],
                random_state=42
            )
            for group in samples_per_group
        ],
        ignore_index=True) # Sample the specified number of images for each group

    print(train_val_df.groupby(['YAS']).size()/len(train_val_df))

    val_set_df = train_val_df.sample(frac=prop_val_of_train_val, random_state=42)
    train_set_df = train_val_df.drop(val_set_df.index)

    # save csvs

    save_dir = os.path.join(save_data_folder, folder_name)
    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'splits'), exist_ok=True)

    train_set_df.to_csv(os.path.join(save_dir,'splits','train.csv'))
    val_set_df.to_csv(os.path.join(save_dir,'splits','val.csv'))
    test_set_df.to_csv(os.path.join(save_dir,'splits','test.csv'))

    # make pkls for images

    splits = ['train','val','test']

    for split in splits:
        meta = pd.read_csv(os.path.join(save_data_folder,folder_name,'splits/{}.csv'.format(split)))
        images = []
        for i in range(len(meta)):
            img = cv2.imread(meta.iloc[i]['Path'],cv2.IMREAD_GRAYSCALE) #so it only has one channel
            # resize to the input size in advance to save time during training
            img = cv2.resize(img, (256, 256))
            images.append(img)
        
        os.makedirs(os.path.join(save_data_folder, folder_name,'pkls'), exist_ok=True)

        with open(os.path.join(save_data_folder,folder_name,'pkls','{}_images.pkl'.format(split)),'wb') as f:
            pickle.dump(images, f)
