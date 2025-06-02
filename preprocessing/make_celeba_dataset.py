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

from utils.make_datasets import *

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CelebA dataset with bias.")
    parser.add_argument("--raw_data_folder", type=str, required=True, help="Path to the raw CelebA data folder (assumes this contains all the images) ")
    parser.add_argument("--attribute_df_path", type=str, required=True, help="Path to the attribute DataFrame (list_attr_celeba.csv)")
    parser.add_argument("--save_data_folder", type=str, required=True, help="Root directory where processed data will be saved")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder name for saving the dataset")
    return parser.parse_args()

if __name__ == "__main__":

    # VARIABLES
    
    args = parse_args()

    raw_data_folder = args.raw_data_folder
    attribute_df_path = args.attribute_df_path
    save_data_folder = args.save_data_folder
    folder_name = args.folder_name

    # PARAMS

    n_images_train = 10000
    prop_images_val = 0.125
    n_images_val = int(n_images_train * prop_images_val)

    size_per_group_test = 312

    prop_positive_images_train_val = 0.5
    prop_male_images_train_val = 0.5
    prop_spurious_images_male_train_val = 0.95
    prop_spurious_images_female_train_val = 0.8

    prop_positive_images_test = 0.5
    prop_male_images_test  = 0.5
    prop_spurious_images_male_test = 0.5
    prop_spurious_images_female_test = 0.5

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

    choices = [0, 1, 2]

    # LOAD DATA

    np.random.seed(42)
    torch.manual_seed(42)

    attribute_df = pd.read_csv(attribute_df_path)
    subset_df = attribute_df[['image_id','Blond_Hair','Male','Smiling']]
    subset_df = subset_df.replace(-1,0)
    subset_df['Female'] = 1 - subset_df['Male']
    subset_df.drop(columns=['Male'],inplace=True)  
    subset_df.columns = ['image_id','Y','S','A']
    subset_df['Path'] = subset_df['image_id'].apply(lambda x: os.path.join(raw_data_folder,x))

    # ADD GROUPS

    AY_map = {
        (0, 0): 0,  # Artefact=0, binaryLabel=0 -> AY=0
        (0, 1): 1,  # Artefact=0, binaryLabel=1 -> AY=1
        (1, 0): 2   # Artefact=1, binaryLabel=0 -> AY=2
    }
    default_AY = 3 # 4th value
    np.random.seed(42)

    subset_df['binary_label'] = subset_df['Y']
    subset_df['AY'] = subset_df.apply(
        lambda row: AY_map.get((row['A'], row['binary_label']), default_AY), axis=1)
    subset_df['SY'] = subset_df.apply(
        lambda row: AY_map.get((row['S'], row['binary_label']), default_AY), axis=1) # same map as AY
    subset_df['AY_8'] = [x if random.random() < 0.5 else x + 4 for x in subset_df['AY']]
    subset_df['SY_8'] = [x if random.random() < 0.5 else x + 4 for x in subset_df['SY']]
    subset_df['Random'] = np.random.choice([0, 1, 2, 3], size=len(subset_df))
    subset_df['Majority'] = [0 if x == 0 or x == 3 else 1 for x in subset_df['AY']]
        
    # Adding noisy columns
    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_AY_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise_with_proportions(subset_df, 'AY', error_percent)['AY']

    conditions = [
        (subset_df['binary_label'] == 0) & (subset_df['A'] == 0) & (subset_df['S'] == 0),
        (subset_df['binary_label'] == 0) & (subset_df['A'] == 0) & (subset_df['S'] == 1),
        (subset_df['binary_label'] == 0) & (subset_df['A'] == 1) & (subset_df['S'] == 0),
        (subset_df['binary_label'] == 0) & (subset_df['A'] == 1) & (subset_df['S'] == 1),
        (subset_df['binary_label'] == 1) & (subset_df['A'] == 0) & (subset_df['S'] == 0),
        (subset_df['binary_label'] == 1) & (subset_df['A'] == 0) & (subset_df['S'] == 1),
        (subset_df['binary_label'] == 1) & (subset_df['A'] == 1) & (subset_df['S'] == 0),
        (subset_df['binary_label'] == 1) & (subset_df['A'] == 1) & (subset_df['S'] == 1)
    ]
    values = [
        '000',
        '001',
        '010',
        '011',
        '100',
        '101',
        '110',
        '111',
    ]

    subset_df['YAS'] = np.select(conditions, values, default=np.nan)

    subset_df['AS'] = subset_df[['A','S']].apply(tuple, axis=1).map(AY_map)

    subset_df['A_4'] = [x if random.random() < 0.5 else x + 2 for x in subset_df['A']]
    subset_df['S_4'] = [x if random.random() < 0.5 else x + 2 for x in subset_df['S']]

    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_A_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise(subset_df, 'A', error_percent)['A']
        
    for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
        error_col = f'noisy_S_{int(error_percent * 100):03}'
        subset_df[error_col] = add_noise(subset_df, 'S', error_percent)['S']
        
    # MAKE TEST DF

    test_set_df = subset_df.groupby('YAS', group_keys=False).apply(lambda x: x.sample(n=size_per_group_test, random_state=42))

    # MAKE TRAIN/VAL DFS
    remaining_df = subset_df.drop(test_set_df.index)
    total_images = n_images_train
    
    samples_per_group = {
        '000': int(total_images * prop_000),
        '001': int(total_images * prop_001),
        '010': int(total_images * prop_010),
        '011': int(total_images * prop_011),
        '110': int(total_images * prop_110),
        '111': int(total_images * prop_111),
        '100': int(total_images * prop_100),
        '101': int(total_images * prop_101),
    }
    for group in samples_per_group: # Ensure no group is oversampled
        available_images = len(remaining_df[remaining_df['YAS'] == group])
        samples_per_group[group] = min(samples_per_group[group], available_images)

    # Sample the specified number of images for each group
    train_val_df = pd.concat(
        [
            remaining_df[remaining_df['YAS'] == group].sample(
                n=samples_per_group[group],
                random_state=42
            )
            for group in samples_per_group
        ],
        ignore_index=True
    )
    val_set_df = train_val_df.sample(frac=prop_val_of_train_val, random_state=42)
    train_set_df = train_val_df.drop(val_set_df.index)

    verify_probabilities(train_set_df,prop_spurious_images_male_train_val,prop_spurious_images_female_train_val)

    # SAVE CSVS

    save_dir = os.path.join(save_data_folder,'splits')
    os.makedirs(save_dir,exist_ok=True)

    train_set_df.to_csv(os.path.join(save_dir,'train.csv'))
    val_set_df.to_csv(os.path.join(save_dir,'val.csv'))
    test_set_df.to_csv(os.path.join(save_dir,'test.csv'))

    # MAKE IMAGES

    splits = ['train','val', 'test']

    for split in splits:
        meta = pd.read_csv(os.path.join(save_dir, f'splits/{split}.csv'))
        images = []
        for i in tqdm(range(len(meta))):
            img = cv2.imread(meta.iloc[i]['Path'], cv2.IMREAD_COLOR)  # read in color (default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
            img = cv2.resize(img, (256, 256))
            images.append(img)

        os.makedirs(os.path.join(save_dir, 'pkls'), exist_ok=True)

        with open(os.path.join(save_dir, 'pkls', f'{split}_images.pkl'), 'wb') as f:
            pickle.dump(images, f)



