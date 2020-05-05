import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.dsa_dataset import DSA_19Dataset

# -------------------------------------------------------
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------


class DSA_19(BaseDataset):
    def __init__(self, opt):
        opt.n_class = 19
        opt.feature_shape = 45
        opt.seq_len = 125
        self.opt = opt
        super(DSA_19, self).__init__(opt)
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

        dataset = DSA_19Dataset(self.path, istrain)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_threads,
            )
     

        return dataset, dataloader


 