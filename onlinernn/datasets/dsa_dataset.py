import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset

# -------------------------------------------------------
# DSA-19 has 4560 training data, 4560 record in test. The first row is label. 
# 125 timesteps per series, 45 input parameters per timestep 5625 features 
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class DSA_19Dataset(Dataset):
    def __init__(self, path, istrain):
        # Normalization to Zero Mean and Unit Standard Deviation
        mu = 0.1656483434051308
        sigma = 3.81595983787948
        # The first column is label
        if istrain:
            datalabels = np.load(path + '/train.npy')
        else:
            datalabels = np.load(path + '/test.npy')
        
        self.data = (datalabels[:, 1:] - mu) / sigma
        # print(self.data.mean())
        # print(self.data.std())
        self.labels = datalabels[:, 0:1]
        # print(np.unique(datalabels[:, 0]))


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.Tensor(self.data[index]).view(125, 45), torch.Tensor(self.labels[index]).long()

        return data, target

