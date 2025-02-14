import torch
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, path_to_images, sens_name, sens_classes, transform):
        super(BaseDataset, self).__init__()
        
        self.dataframe = dataframe        
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.path_to_images = path_to_images
        self.sens_name = sens_name
        self.sens_classes = sens_classes
        
        self.A = None
        self.Y = None
        self.AY_proportion = None
        
    def get_AY_proportions(self):
        if self.AY_proportion:
            return self.AY_proportion
        
        A_num_class = 2
        Y_num_class = 2
        A_label = self.A
        Y_label = self.Y
        
        A = self.A.tolist()
        Y = self.Y.tolist()
        ttl = len(A)
            
        len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
        len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
        len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
        len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

        assert (
            len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1
        ) == ttl, "Problem computing train set AY proportion."
        A0Y0 = len_A0Y0 / ttl
        A0Y1 = len_A0Y1 / ttl
        A1Y0 = len_A1Y0 / ttl
        A1Y1 = len_A1Y1 / ttl
        
        self.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]
        
        return self.AY_proportion
    
    def get_A_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def get_Y_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def set_A(self, sens_name):
        if sens_name == 'Sex':
            A = np.asarray(self.dataframe['Sex'].values != 'M').astype('float')
        elif sens_name == 'Age':
            if self.sens_classes == 4:
                A = np.asarray(self.dataframe['Age_multi4'].values.astype('float'))
            else:
                A = np.asarray(self.dataframe['Age_binary'].values.astype('int') == 1).astype('float')
        elif sens_name == 'Race':
            A = np.asarray(self.dataframe['Race'].values == 'White').astype('float')
        elif self.sens_name == 'skin_type':
            A = np.asarray(self.dataframe['skin_binary'].values != 0).astype('float')
        elif self.sens_name == 'Insurance':
            self.A = np.asarray(self.dataframe['Insurance_binary'].values != 0).astype('float')
        elif self.sens_name == 'Ethnicity': # added extra attribute (not binary, need to check if OK)
            A = np.asarray(self.dataframe['Ethnicity'].values.astype('float'))
        elif self.sens_name == 'Centre': # added extra attribute (not binary, need to check if OK)
            A = np.asarray(self.dataframe['Centre'].values.astype('float'))
        elif self.sens_name == 'Random':
            A = np.asarray(self.dataframe['Random'].values.astype('float'))
        elif self.sens_name == 'noisy_001' or self.sens_name == 'noisy_005' or self.sens_name == 'noisy_010' or self.sens_name == 'noisy_025' or self.sens_name == 'noisy_050' or self.sens_name == 'Age_multi8' or self.sens_name == 'Random' or self.sens_name == 'Y' or self.sens_name == 'Majority' or self.sens_name == 'Imperfect' or self.sens_name == 'Age_multi4' or self.sens_name == 'Imperfect_paper' or self.sens_name == 'AY' or self.sens_name == 'SY':
            A = np.asarray(self.dataframe[self.sens_name].values.astype('float'))
        else:
            A = np.asarray(self.dataframe[self.sens_name].values.astype('float'))
        # else:
        #     raise ValueError("Does not contain {}".format(self.sens_name))
        return A

    def get_weights(self, resample_which):
        sens_attr, group_num = self.group_counts(resample_which) #sens_attr is array of sens_attr membership for each sample
        group_weights = [1/x.item() for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights
    
    def group_counts(self, resample_which = 'group'):
        if resample_which == 'group' or resample_which == 'balanced':
            if self.sens_name == 'Sex':
                mapping = {'M': 0, 'F': 1}
                groups = self.dataframe['Sex'].values
                group_array = [*map(mapping.get, groups)]
                
            elif self.sens_name == 'Age':
                if self.sens_classes == 2:
                    groups = self.dataframe['Age_binary'].values
                elif self.sens_classes == 5:
                    groups = self.dataframe['Age_multi'].values
                elif self.sens_classes == 4:
                    groups = self.dataframe['Age_multi4'].values.astype('int')
                group_array = groups.tolist()
                
            elif self.sens_name == 'Race':
                mapping = {'White': 0, 'non-White': 1}
                groups = self.dataframe['Race'].values
                group_array = [*map(mapping.get, groups)]
            elif self.sens_name == 'skin_type':
                if self.sens_classes == 2:
                    groups = self.dataframe['skin_binary'].values
                elif self.sens_classes == 6:
                    groups = self.dataframe['skin_type'].values
                group_array = groups.tolist()
            elif self.sens_name == 'Insurance':
                if self.sens_classes == 2:
                    groups = self.dataframe['Insurance_binary'].values
                elif self.sens_classes == 5:
                    groups = self.dataframe['Insurance'].values
                group_array = groups.tolist()
            
            elif self.sens_name == 'Ethnicity':
                if self.sens_classes == 4:
                    groups = self.dataframe['Ethnicity'].values.astype('int')
                    group_array = groups.tolist()
            elif self.sens_name == 'Centre':
                if self.sens_classes == 6 or self.sens_classes == 5:
                    groups = self.dataframe['Centre'].values.astype('int')
                    group_array = groups.tolist()
            elif self.sens_name == 'Random':
                if self.sens_classes == 4:
                    groups = self.dataframe['Random'].values.astype('int')
                    group_array = groups.tolist()    

            elif self.sens_name == 'noisy_001' or self.sens_name == 'noisy_005' or self.sens_name == 'noisy_010' or self.sens_name == 'noisy_025' or self.sens_name == 'noisy_050' or self.sens_name == 'Age_multi4':
                if self.sens_classes == 4:
                    groups = self.dataframe[self.sens_name].values.astype('int')
                    group_array = groups.tolist()
            
            elif self.sens_name == 'Age_multi8' or self.sens_name == 'Random' or self.sens_name == 'Y' or self.sens_name == 'Majority' or self.sens_name == 'Imperfect' or self.sens_name == 'Imperfect_paper' or self.sens_name == 'AY' or self.sens_name == 'SY':
                groups = self.dataframe[self.sens_name].values.astype('int')
                group_array = groups.tolist()
            
            else:
                groups = self.dataframe[self.sens_name].values.astype('int')
                group_array = groups.tolist()             

            # else:
            #     raise ValueError("sensitive attribute does not defined in BaseDataset")
            
            if resample_which == 'balanced':
                #get class
                labels = self.Y.tolist()
                num_labels = len(set(labels))
                num_groups = len(set(group_array))
                
                group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()
                
        elif resample_which == 'class':
            group_array = self.Y.tolist()
            num_labels = len(set(group_array))
        
        self._group_array = torch.LongTensor(group_array)
        if resample_which == 'group':
            self._group_counts = (torch.arange(self.sens_classes).unsqueeze(1)==self._group_array).sum(1).float() # tensor with group counts for each sens attribute class
        elif resample_which == 'balanced':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'class':
            self._group_counts = (torch.arange(num_labels).unsqueeze(1)==self._group_array).sum(1).float()
        return group_array, self._group_counts
    
    def __len__(self):
        return self.dataset_size
    
    def get_labels(self): 
        # for sensitive attribute imbalance
        if self.sens_classes == 2:
            if self.sens_name == 'Y' or self.sens_name == 'Majority' or self.sens_name == 'Age':
                return self.dataframe[self.sens_name].values.tolist()
            else:
                #return self.A
                return self.dataframe[self.sens_name].values.tolist()
        elif self.sens_classes == 5:
            if self.sens_name == 'Age':
                return self.dataframe['Age_multi'].values.tolist()
            elif self.sens_name == 'Centre':
                return self.dataframe['Centre'].values.tolist()
            else:
                return self.dataframe[self.sens_name].values.tolist()
        elif self.sens_classes == 4:
            if self.sens_name == 'Age':
                return self.dataframe['Age_multi4'].values.tolist()
            elif self.sens_name == 'Ethnicity': 
                return self.dataframe['Ethnicity'].values.tolist()
            elif self.sens_name == 'noisy_001' or self.sens_name == 'noisy_005' or self.sens_name == 'noisy_010' or self.sens_name == 'noisy_025' or self.sens_name == 'noisy_050' or self.sens_name == 'Random' or self.sens_name == 'Imperfect' or self.sens_name == 'Age_multi4' or self.sens_name == 'AY' or self.sens_name == 'SY':
                return self.dataframe[self.sens_name].values.tolist()
            else:
                return self.dataframe[self.sens_name].values.tolist()
        elif self.sens_classes == 6:
            if self.sens_name == 'Centre':
                return self.dataframe['Centre'].values.tolist()
            else:
                return self.dataframe[self.sens_name].values.tolist()
        elif self.sens_classes == 8:
            if self.sens_name == 'Age_multi8':
                return self.dataframe['Age_multi8'].values.tolist()
            else:
                return self.dataframe[self.sens_name].values.tolist()
        else:
            return self.dataframe[self.sens_name].values.tolist()

    def get_sensitive(self, sens_name, sens_classes, item):
        if sens_name == 'Sex':
            if item['Sex'] == 'M':
                sensitive = 0
            else:
                sensitive = 1
        elif sens_name == 'Age':
            if sens_classes == 2:
                sensitive = int(item['Age_binary'])
            elif sens_classes == 5:
                sensitive = int(item['Age_multi'])
            elif sens_classes == 4:
                sensitive = int(item['Age_multi4'])
        elif sens_name == 'Race':
            if item['Race'] == 'White':
                sensitive = 0
            else:
                sensitive = 1
        elif sens_name == 'skin_type':
            if sens_classes == 2:
                sensitive = int(item['skin_binary'])
            else:
                sensitive = int(item['skin_type'])
        elif self.sens_name == 'Insurance':
            if self.sens_classes == 2:
                sensitive = int(item['Insurance_binary'])
            elif self.sens_classes == 5:
                sensitive = int(item['Insurance'])
        elif self.sens_name == 'Ethnicity':
            if sens_classes == 4:
                sensitive = int(item['Ethnicity'])
        elif self.sens_name == 'Centre':
            if sens_classes == 6:
                sensitive = int(item['Centre'])
            elif sens_classes == 5:
                sensitive = int(item['Centre'])
        elif self.sens_name == 'Random':
            if sens_classes == 4:
                sensitive = int(item['Random'])
        elif self.sens_name == 'noisy_001' or self.sens_name == 'noisy_005' or self.sens_name == 'noisy_010' or self.sens_name == 'noisy_025' or self.sens_name == 'noisy_050' or self.sens_name == 'Age_multi8' or self.sens_name == 'Random' or self.sens_name == 'Y' or self.sens_name == 'Majority' or self.sens_name == 'Imperfect' or self.sens_name == 'Age_multi4' or self.sens_name == 'Imperfect_paper' or self.sens_name == 'AY' or self.sens_name == 'SY':
            # assume correct number of classes have been defined
            sensitive = int(item[self.sens_name])
        else:
            sensitive = int(item[self.sens_name])
        # else:
        #     raise ValueError('Please check the sensitive attributes.')
        return sensitive