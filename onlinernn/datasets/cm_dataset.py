import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset

# -------------------------------------------------------
# Ref: https://github.com/rand0musername/urnn/tree/master/problems
# https://github.com/zhangjiong724/spectral-RNN/blob/master/code/rnn.py
# -------------------------------------------------------
np.random.seed(42)

class CM_Dataset(Dataset):
    def __init__(self, istrain, N_SAMPLES, N_TIMESTEPS, N_TRAIN, transform=None):
        self.transform = transform
        X, Y = generate_copy_task_data(N_SAMPLES, N_TIMESTEPS)
      
        if istrain:
            self.data, self.labels = X[:N_TRAIN], Y[:N_TRAIN]
        else:
            self.data, self.labels = X[N_TRAIN:], Y[N_TRAIN:]

      
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data, target = torch.Tensor(self.data[index]), torch.Tensor(self.labels[index]).long()

        if self.transform:
            data = self.transform(data)
        return data, target

def generate_copy_task_data(num_samples, sample_len):
    assert(sample_len > 20) # must be

    X = np.zeros((num_samples, sample_len, 1))
    data = np.random.randint(low = 1, high = 9, size = (num_samples, 10, 1))
   
    X[:, :10] = data
    X[:, -11] = 9
    Y = np.zeros((num_samples, sample_len, 1))
    Y[:, -10:] = X[:, :10]
    return X, Y


