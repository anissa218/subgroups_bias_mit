import torch
import pickle
import numpy as np
import pickle
from datasets.BaseDataset import BaseDataset
from transformers import BertTokenizer

class CIVILCOMMENTS(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform,subsample_what = None):
        super(CIVILCOMMENTS, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform,subsample_what)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['binary_label'].values) > 0).astype('float')
        self.AY_proportion = None

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        text = item['comment_text']
        label = torch.FloatTensor([int(item['toxicity'].astype('float') >= 0.5)])
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)

        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return (input_ids, attention_mask), label, sensitive, idx

