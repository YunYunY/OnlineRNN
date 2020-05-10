import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.mnist_byte_dataset import MNIST_byte_Dataset
from onlinernn.datasets.data_utils import AddGaussianNoise

# -------------------------------------------------------
# Load MNIST data from scratch
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class MNIST_byte(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.shuffle = opt.shuffle
        opt.n_class = 10
        opt.feature_shape = 1
        opt.seq_len = 784
        super(MNIST_byte, self).__init__(opt)
        istrain = (opt.istrain or opt.continue_train)
        self.dataset = self.torch_loader(istrain=istrain)
        self.dataset, self.dataloader = self.torch_loader(istrain=istrain)
        print(f"Total datasize is {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
        
    # ----------------------------------------------
    def torch_loader(self, istrain):
        """
            Fetch data by torch.utils.data.Dataset
            Create dataloader
            Args:
                istrain: flag condition for getting training or test data
        """
        
        if self.opt.add_noise:
            transform=transforms.Compose([
                    AddGaussianNoise(0., 2.)
                ])
        else:
            transform = None 
        dataset = MNIST_byte_Dataset(self.path, istrain, transform)
        print(dataset[0][0][211])
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_threads,
            )
     

        return dataset, dataloader

  # ----------------------------------------------
    def subset_loader(self, i):
        """
            Fetch data by torch.utils.data.Subset
            Create dataloader
            Args:
                i: index 
        """
        sampler = SubsetRandomSampler(list(range(i, i + self.batch_size)))
        train_loader = torch.utils.data.DataLoader(self.dataset,
                sampler=sampler, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_threads, drop_last=True)
        return train_loader