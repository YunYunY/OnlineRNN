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

        # The first column is label
        if istrain:
            self.datalabels = np.load(path + '/train.npy')
        else:
            self.datalabels = np.load(path + '/test.npy')
        # print(np.unique(datalabels[:, 0]))


    def __len__(self):
        return len(self.datalabels)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.Tensor(self.datalabels[index, 1:]).view(-1, 1), torch.Tensor(self.datalabels[index, 0:1]).long()
        return data, target

