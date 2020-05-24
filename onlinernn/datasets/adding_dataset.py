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

class Adding_Dataset(Dataset):
    def __init__(self, istrain, N_SAMPLES, N_TIMESTEPS, N_TRAIN, transform=None):
        self.transform = transform
        X, Y = generate_add_task_data(N_SAMPLES, N_TIMESTEPS)
      
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

        data, target = torch.Tensor(self.data[index]), torch.Tensor(self.labels[index])

        if self.transform:
            data = self.transform(data)
        return data, target


def generate_add_task_data(num_samples, sample_len):
  
    X_value = np.random.uniform(low = 0, high = 1, size = (num_samples, sample_len, 1))
    X_mask = np.zeros((num_samples, sample_len, 1))

    Y = np.ones((num_samples, 1))
    
    for i in range(num_samples):
        half = int(sample_len / 2)
        first_i = np.random.randint(half)
        second_i = np.random.randint(half) + half
        X_mask[i, (first_i, second_i), 0] = 1
        Y[i, 0] = np.sum(X_value[i, (first_i, second_i), 0])
    X = np.concatenate((X_value, X_mask), 2)
    return X, Y

    