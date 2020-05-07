import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset

# -------------------------------------------------------
# Custom HAR-2 data has 7352 training series (with 50% overlap between each serie), 2947 record in test. The first row is label. 
# 128 timesteps per series, 9 input parameters per timestep, 1152 features
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class HAR_2Dataset(Dataset):
    def __init__(self, path, istrain, transform=None):
        self.transform = transform
        # Normalization to Zero Mean and Unit Standard Deviation
        mu = 0.10206605722975093
        sigma = 0.4021651763839265
        # self.slice_interval = 8 
        # The first column is label
        if istrain:
            datalabels = np.load(path + '/train.npy')
        else:
            datalabels = np.load(path + '/test.npy')

        self.data = (datalabels[:, 1:] - mu) / sigma
        # print(self.data[0], sep='\n')

        self.labels = datalabels[:, 0:1]
        # print(np.unique(datalabels[:, 0]))
        # dim0, dim1 = self.data.shape[0], self.data.shape[1]
        # if slice:
        #     print('Slice data')
        #     result = np.zeros(shape=(int(dim0 * (dim1/9/self.slice_interval)), 9*self.slice_interval))
        #     result_labels = np.zeros(shape=(int(dim0 * (dim1/9/self.slice_interval)), 1))
        #     count = 0
        #     for i in range(dim0):
        #         for j in range(0, dim1, 9 * self.slice_interval):
        #             result[count, :] = self.data[i, j:j+9*self.slice_interval]
        #             result_labels[count, :] = self.labels[i, :]
        #             count += 1
        #     self.data = result
        #     self.labels = result_labels
        




    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.slice:
            # data, target = torch.Tensor(self.data[index]).view(self.slice_interval, 9), torch.Tensor(self.labels[index]).long()

        # else:

        data, target = torch.Tensor(self.data[index]).view(128, 9), torch.Tensor(self.labels[index]).long()
        if self.transform:
            data = self.transform(data)
        return data, target

