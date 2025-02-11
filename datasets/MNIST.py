import torch
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset
import cv2

class MNIST(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform,use_pkl=True):
        super(MNIST, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        """
            Dataset class for MNIST images.
            
            Arguments:
            dataframe: the metadata in pandas dataframe format.
            path_to_pickles: path to the pickle file containing images.
            sens_name: which sensitive attribute to use, e.g., Sex or Age_binary or Ethnicity(others not defined, and Age_multi has 3 classes instead of 4)
            sens_classes: number of sensitive classes. Depends on attribute
            transform: whether conduct data transform to the images or not.
            
            Returns:
            index, image, label, and sensitive attribute.
        """

        self.tol_images = torch.load(path_to_pickles).numpy()
        #print(self.tol_images.shape)
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['binaryLabel'].values)).astype('float')
        self.AY_proportion = None
        self.use_pkl = use_pkl
    
    def __getitem__(self, idx):
        # get the item based on the index
        item = self.dataframe.iloc[idx]

        #img = Image.fromarray(self.tol_images[idx].astype(np.uint8)) #.convert('RGB') 
        img = self.tol_images[idx].astype(np.uint8)
        img = img.transpose(1,2,0) # because transform toTensor() assumes shape is (H, W, C) and changes it to (C,H,W) but this is not the cse in the dataset
    
        img = self.transform(img)
        
        label = torch.FloatTensor([int(item['binaryLabel'])])
        
        # get sensitive attributes in numerical values
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return img, label, sensitive, idx