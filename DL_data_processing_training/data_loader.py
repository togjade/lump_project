#!/usr/bin/python
import numpy as np
import torch
from torch.utils.data import Dataset


class lumpDataset(Dataset):
    def __init__(self, x, y, transform=False):
        super(lumpDataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    ################################################################
    def __get_item__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # read feature
        features = self.x[idx, :, :]
        # read label
        labels = self.y[idx, :]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 2)
        sample = {'features':features, 'labels':labels}
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        x_shape = self.x.shape
        return x_shape[0]
    
class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, labels = sample['features'], sample['labels']

        return {'features': torch.from_numpy(features),
                'labels': torch.from_numpy(labels),
               }
