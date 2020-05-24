import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.adding_dataset import Adding_Dataset

# -------------------------------------------------------
# Build adding task, time seq_len is user defined, each data has (seq_len, 2) features 
# -------------------------------------------------------


class ADDING(BaseDataset):
    def __init__(self, opt):
        self.N_TRAIN = opt.adding_train#100000
        self.N_TEST = opt.adding_test #1000
        self.N_SAMPLES = self.N_TEST + self.N_TRAIN

        opt.n_class = 1
        opt.feature_shape = 2
        self.opt = opt
        super(ADDING, self).__init__(opt)
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

        dataset = Adding_Dataset(istrain, self.N_SAMPLES, self.opt.seq_len, self.N_TRAIN)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_threads,
            )
     

        return dataset, dataloader


 