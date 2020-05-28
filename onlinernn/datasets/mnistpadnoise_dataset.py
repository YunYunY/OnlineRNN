import numpy as np
import torch
import gzip
from torch.utils.data import Dataset, DataLoader
from onlinernn.datasets.base_dataset import BaseDataset

# -------------------------------------------------------
# -------------------------------------------------------
torch.set_printoptions(precision=8)

class MNIST_Pad_Noise_Dataset(Dataset):

    def __init__(self, path, istrain, transform=None):
        self.transform = transform
        rand_mean=0.
        rand_std=1.
       
        if istrain:
            data_filename = path + '/raw/train-images-idx3-ubyte.gz'
            label_filename = path + '/raw/train-labels-idx1-ubyte.gz'
        else:
            data_filename = path + '/raw/t10k-images-idx3-ubyte.gz'
            label_filename = path + '/raw/t10k-labels-idx1-ubyte.gz'

        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(data_filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)

        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        data = data / np.float32(256)

     
        # data = data.reshape(data.shape[0], -1, 1)

        self.data = data.astype('float32')

        # scale to [-1, 1]
        # self.data -= 0.5
        # self.data *= 2
        self.data = (self.data-0.13015018)/0.30690408 # zero mean, unit variance
        np.random.seed(42)
        # extend data dimension and pad with random noise
        noise = torch.randn((self.data.shape[0], 1, 1000, self.data.shape[2])) * rand_std + rand_mean
        noise[:, :, 0:28, :] =  torch.Tensor(self.data)
        self.data = noise

        with gzip.open(label_filename, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
        self.labels = self.labels.astype(np.int32)


            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # data, target = torch.Tensor(self.data[index]), torch.from_numpy(np.asarray(self.labels[index])).long()
        data, target = self.data[index], torch.from_numpy(np.asarray(self.labels[index])).long()


        if self.transform:
            data = self.transform(data)
        return data, target

