import pandas as pd
import numpy as np
import torch

def assign_artefact(row,prop_spurious_male,prop_spurious_female):
    '''
    assign spurious artefact to male and female images according to the proportions specified
    assumes that the proportion of spurious correlation is the same for positive and negative images!
    '''
    if row['binaryLabel'] == 1:  # Odd label
        if row['Sex'] == 'M':
            return np.random.choice([1, 0], p=[prop_spurious_male, 1-prop_spurious_male])
        else:  # Sex is 'F'
            return np.random.choice([1, 0], p=[prop_spurious_female, 1-prop_spurious_female])
    else:  # Even label
        if row['Sex'] == 'M':
            return np.random.choice([0, 1], p=[prop_spurious_male, 1-prop_spurious_male])
        else:  # Sex is 'F'
            return np.random.choice([0, 1], p=[prop_spurious_female, 1-prop_spurious_female])

def assign_from_mapping(row, condition_map, default_value):
    # Extract relevant keys for mapping
    key = tuple(row[column] for column in condition_map.keys())
    return condition_map.get(key, default_value)

def verify_probabilities(df,prop_spurious_male,prop_spurious_female):
    results = {}
    
    # Odd labels and Male
    male_odd = df[(df['binaryLabel'] == 1) & (df['Sex'] == 'M')]
    results['Male Odd (1 expected ~' + str(100*prop_spurious_male) + '%)'] = male_odd['Artefact'].mean()
    
    # Odd labels and Female
    female_odd = df[(df['binaryLabel'] == 1) & (df['Sex'] == 'F')]
    results['Female Odd (1 expected ~' + str(100*prop_spurious_female) + '%)'] = female_odd['Artefact'].mean()
    
    # Even labels and Male
    male_even = df[(df['binaryLabel'] == 0) & (df['Sex'] == 'M')]
    results['Male Even (0 expected ~' + str(100*prop_spurious_male) + '%)'] = 1 - male_even['Artefact'].mean()
    
    # Even labels and Female
    female_even = df[(df['binaryLabel'] == 0) & (df['Sex'] == 'F')]
    results['Female Even (0 expected ~' + str(100*prop_spurious_female) + '%)'] = 1 - female_even['Artefact'].mean()
    
    return results

def add_noise(df, col, error_percent):
    '''
    mislabel a percentage of the data in the specified column by randomly sampling from any of the possible values (including the current one)
    '''
    num_changes = int(len(df) * error_percent)
    change_indices = np.random.choice(df.index, size=num_changes, replace=False)
    n_unique = len(df[col].unique()) # either 2 or 4 depending on how many possible values
    new_values = [np.random.choice([x for x in range(n_unique) if x != df.loc[i, col]]) for i in change_indices]
    
    df_copy = df.copy()
    df_copy.loc[change_indices, col] = new_values
    
    return df_copy

def add_noise_with_proportions(df, col, error_percent):
    '''
    same as add_noise except instead of doing uniform random sampling from all possible values, it samples from the current distribution of values (so that the biased distribution stays the same with noise)
    '''
    # Calculate the probabilities of each value in the column
    value_counts = df[col].value_counts(normalize=True)  # Get normalized counts
    values = value_counts.index
    probabilities = value_counts.values

    # Determine the number of changes
    num_changes = int(len(df) * error_percent)

    # Select indices to change
    change_indices = np.random.choice(df.index, size=num_changes, replace=False)

    # Generate new values for those indices based on the original proportions (can be the original value)
    new_values = np.random.choice(values, size=num_changes, p=probabilities)

    df_copy = df.copy()
    df_copy.loc[change_indices, col] = new_values

    return df_copy


def add_colour_fg_bg(img_tensor,colour_channel,bg_colour_channel,noise=1):
    '''
    img_tensor: torch.tensor with values between 0 and 255 of dim [28,28]
    colour_channel: int, colour channel to add (can be multiple), either 0 or 1
    noise: how much to add colours from other channels
    ** only for 0s and 1s colour channel
    returns: 3 channel image tensor [3,H,W]
    '''
    if bg_colour_channel == 0:
        rgb_imgs = torch.zeros(3,img_tensor.shape[0],img_tensor.shape[1],dtype=img_tensor.dtype)
    elif bg_colour_channel == 1:
        rgb_imgs = torch.ones(3,img_tensor.shape[0],img_tensor.shape[1],dtype=img_tensor.dtype)*255

    mask = img_tensor > 0  # Only modify where img_tensor has non-zero values
    rgb_imgs[colour_channel, :, :][mask] = img_tensor[mask]
    rgb_imgs[(colour_channel + 1) % 3, :, :][mask] = (img_tensor[mask] * torch.rand(mask.sum()) * noise).to(img_tensor.dtype) # randomly add a bit of other colours (not to background)
    rgb_imgs[(colour_channel + 2) % 3, :, :][mask] = (img_tensor[mask] * torch.rand(mask.sum()) * noise).to(img_tensor.dtype)

    return rgb_imgs
    
def make_subset_images(images_df,images_tensor,fg_colour_channel_col='Sex_binary',bg_colour_channel_col = 'Artefact',noise=1):
    '''
    images_df: df with col 'image_index' and 'image_colour_channel'
    images_tensor: [n_images,28,28] tensor
    return tensor of all images with colour: [n_images,3,28,28]
    '''

    subset_images = torch.zeros(len(images_df),3,images_tensor.shape[1],images_tensor.shape[2])
    for i in range(len(images_df)):
        index = images_df.iloc[i]['image_index']
        fg_colour_channel = images_df.iloc[i][fg_colour_channel_col]
        bg_colour_channel = images_df.iloc[i][bg_colour_channel_col]
        rgb_image = add_colour_fg_bg(images_tensor[index],fg_colour_channel,bg_colour_channel,noise)
        subset_images[i] = rgb_image
    return subset_images