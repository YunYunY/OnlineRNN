import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.har_dataset import HAR_2Dataset
# -------------------------------------------------------
# HAR-2 data has 7352 in training, 2947 in test. The first row is label. 
# Ref: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------

class HAR_2(BaseDataset):
    def __init__(self, opt):
        opt.n_class = 2
        opt.feature_shape = 1
        opt.seq_len = 1152
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


        dataset = HAR_2Dataset(self.path, istrain)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_threads,
            )

        return dataset, dataloader

