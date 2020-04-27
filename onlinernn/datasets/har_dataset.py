import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset

# -------------------------------------------------------
# Custom HAR-2 data has 7352 record in training, 2947 record in test. The first row is label. 
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class HAR_2Dataset(Dataset):
    def __init__(self, path, istrain):
        # Normalization to Zero Mean and Unit Standard Deviation
        mu = 0.10206605722975093
        sigma = 0.4021651763839265
        # The first column is label
        if istrain:
            datalabels = np.load(path + '/train.npy')
        else:
            datalabels = np.load(path + '/test.npy')
        self.data = (datalabels[:, 1:] - mu) / sigma

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
        data, target = torch.Tensor(self.data[index]).view(-1, 1), torch.Tensor(self.labels[index]).long()

        return data, target

