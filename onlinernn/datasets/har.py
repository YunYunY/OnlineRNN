import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.har_dataset import HAR_2Dataset
from onlinernn.datasets.data_utils import AddGaussianNoise

# -------------------------------------------------------
# HAR-2 data has 7352 in training, 2947 in test. The first row is label. 
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class HAR_2(BaseDataset):
    def __init__(self, opt):
        opt.n_class = 2
        opt.feature_shape = 9
        opt.seq_len = 128
        self.opt = opt
        super(HAR_2, self).__init__(opt)
        istrain = (opt.istrain or opt.continue_train)
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

        # dataset = HAR_2Dataset(self.path, istrain, self.opt.slice)
        
        if self.opt.add_noise:
            transform=transforms.Compose([
                    AddGaussianNoise(0., 2.)
                ])
        else:
            transform = None 
        dataset = HAR_2Dataset(self.path, istrain, transform)

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