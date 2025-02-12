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
    parser = argparse.ArgumentParser(description="Generate MNIST dataset with bias.")
    parser.add_argument("--raw_data_folder", type=str, required=True, help="Path to the raw MNIST data folder (assumes this contains a 'training.pt' and 'test.pt' file) ")
    parser.add_argument("--save_data_folder", type=str, required=True, help="Root directory where processed data will be saved")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder name for saving the dataset")
    return parser.parse_args()

if __name__ == "__main__":

    # VARIABLES
    
    args = parse_args()

    raw_data_folder = args.raw_data_folder
    save_data_folder = args.save_data_folder
    folder_name = args.folder_name

    # PARAMS

    n_images_train = 5000

    prop_images_val = 0.125
    n_images_val = int(n_images_train * prop_images_val)

    prop_images_test = 0.25
    n_images_test = int(n_images_train * prop_images_test)

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

    # Load data

    np.random.seed(42)
    torch.manual_seed(42)

    train_data = torch.load(os.path.join(raw_data_folder, 'training.pt'))
    test_data = torch.load(os.path.join(raw_data_folder, 'test.pt'))

    # Separate images and labels
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    # Make DFs

    train_images_df = pd.DataFrame()
    index_array = np.linspace(0,len(train_images)-1,len(train_images),dtype=int)
    train_images_df['image_index'] = index_array
    train_images_df['image_label'] = train_labels
    train_images_df['binaryLabel'] = train_images_df['image_label']%2 # 0 if even, 1 if odd

    np.random.seed(42)
    subset_train_val_images_df = train_images_df.sample(n=n_images_train+n_images_val, random_state=42)
    np.random.seed(42)
    subset_train_val_images_df['Sex'] = np.random.choice(
        ['M', 'F'], 
        size=n_images_train + n_images_val, 
        p=[prop_male_images_train_val, 1-prop_male_images_train_val],  # Probabilities for 'M' and 'F'
    )
    subset_train_val_images_df['Sex_binary'] = subset_train_val_images_df['Sex'].apply(lambda x: 0 if x == 'M' else 1)
    subset_train_val_images_df['Artefact'] = subset_train_val_images_df.apply(assign_artefact, axis=1,prop_spurious_male = prop_spurious_images_male_train_val,prop_spurious_female=prop_spurious_images_female_train_val)
    subset_train_val_images_df['AY'] = subset_train_val_images_df.apply(
        lambda row: AY_map.get((row['Artefact'], row['binaryLabel']), default_AY), axis=1
    )
    subset_train_val_images_df['SY'] = subset_train_val_images_df.apply(
        lambda row: SY_map.get((row['Sex'], row['binaryLabel']), default_SY), axis=1
    )

    # sanity check
    verification_results = verify_probabilities(subset_train_val_images_df,prop_spurious_images_male_train_val,prop_spurious_images_female_train_val)
    for condition, prob in verification_results.items():
        print(f"{condition}: {prob:.2%}")
    subset_train_val_images_df.groupby(['Sex'])['AY'].value_counts()

    subset_train_images_df = subset_train_val_images_df.iloc[:n_images_train]
    subset_val_images_df = subset_train_val_images_df.iloc[n_images_train:]

    test_images_df = pd.DataFrame()
    index_array = np.linspace(0,len(test_images)-1,len(test_images),dtype=int)
    test_images_df['image_index'] = index_array
    test_images_df['image_label'] = test_labels
    test_images_df['binaryLabel'] = test_images_df['image_label']%2 # 0 if even, 1 if odd

    # Test set
    np.random.seed(42)
    subset_test_images_df = test_images_df.sample(n=n_images_test, random_state=42)
    np.random.seed(42)
    subset_test_images_df['Sex'] = np.random.choice(
        ['M', 'F'], 
        size=n_images_test, 
        p=[prop_male_images_test, 1-prop_male_images_test],  # Probabilities for 'M' and 'F'
    )
    subset_test_images_df['Sex_binary'] = subset_test_images_df['Sex'].apply(lambda x: 0 if x == 'M' else 1)
    subset_test_images_df['Artefact'] = subset_test_images_df.apply(assign_artefact, axis=1,prop_spurious_male = prop_spurious_images_male_test,prop_spurious_female=prop_spurious_images_female_test)
    subset_test_images_df['AY'] = subset_test_images_df.apply(
        lambda row: AY_map.get((row['Artefact'], row['binaryLabel']), default_AY), axis=1
    )
    subset_test_images_df['SY'] = subset_test_images_df.apply(
        lambda row: SY_map.get((row['Sex'], row['binaryLabel']), default_SY), axis=1
    )

    # sanity check
    verification_results = verify_probabilities(subset_test_images_df,prop_spurious_images_male_test,prop_spurious_images_female_test)
    for condition, prob in verification_results.items():
        print(f"{condition}: {prob:.2%}")
    subset_test_images_df.groupby(['Sex'])['AY'].value_counts()

    np.random.seed(42)

    dfs = [subset_train_images_df, subset_val_images_df, subset_test_images_df]

    for df in dfs:
        df['Y'] = df['binaryLabel']
        df['AY_8'] = [x if random.random() < 0.5 else x + 4 for x in df['AY']]
        df['SY_8'] = [x if random.random() < 0.5 else x + 4 for x in df['SY']]
        df['Random'] = np.random.choice([0, 1, 2, 3], size=len(df))
        df['Majority'] = [0 if x == 0 or x == 3 else 1 for x in df['AY']]
        df['YAS'] = df[['Y','Artefact','Sex_binary']].apply(tuple, axis=1).map(YAS_map)
        df['AS'] = df[['Artefact','Sex_binary']].apply(tuple, axis=1).map(AS_map)
        df['A_4'] = [x if random.random() < 0.5 else x + 2 for x in df['Artefact']]
        df['S_4'] = [x if random.random() < 0.5 else x + 2 for x in df['Sex_binary']]

        for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
            error_col = f'noisy_A_{int(error_percent * 100):03}'
            df[error_col] = add_noise(df, 'Artefact', error_percent)['Artefact']
        
        for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
            error_col = f'noisy_S_{int(error_percent * 100):03}'
            df[error_col] = add_noise(df, 'Sex_binary', error_percent)['Sex_binary']

        # Adding noisy columns
        for error_percent in [0.01, 0.05, 0.1, 0.25, 0.5]:
            error_col = f'noisy_AY_{int(error_percent * 100):03}'
            df[error_col] = add_noise_with_proportions(df, 'AY', error_percent)['AY']

    # Make images
    subset_train_images = make_subset_images(subset_train_images_df,train_images,fg_colour_channel_col='Sex_binary',bg_colour_channel_col = 'Artefact',noise=1)
    subset_val_images = make_subset_images(subset_val_images_df,train_images,fg_colour_channel_col='Sex_binary',bg_colour_channel_col = 'Artefact',noise=1)
    subset_test_images = make_subset_images(subset_test_images_df,test_images,fg_colour_channel_col='Sex_binary',bg_colour_channel_col = 'Artefact',noise=1)

    os.makedirs(os.path.join(save_data_folder, folder_name),exist_ok=True)
    os.makedirs(os.path.join(save_data_folder, folder_name,'pkls'),exist_ok=True)
    os.makedirs(os.path.join(save_data_folder, folder_name,'splits'),exist_ok=True)

    torch.save(subset_train_images.int(), os.path.join(save_data_folder, folder_name,'pkls','train_images.pt'))
    torch.save(subset_val_images.int(), os.path.join(save_data_folder, folder_name,'pkls','val_images.pt'))
    torch.save(subset_test_images.int(), os.path.join(save_data_folder, folder_name,'pkls','test_images.pt'))

    subset_train_images_df.to_csv(os.path.join(save_data_folder, folder_name, 'splits','train.csv'))
    subset_val_images_df.to_csv(os.path.join(save_data_folder, folder_name, 'splits','val.csv'))
    subset_test_images_df.to_csv(os.path.join(save_data_folder, folder_name, 'splits','test.csv'))
