from abc import ABC
import os
import torchvision
import torch



class BaseDataset(ABC):
    """
        Abstract class defining the APIs for dataset classes
    """

    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.num_threads = opt.num_threads
        self._n_class = opt.n_class

    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
    
        return self.class_name()

    @property
    def path(self):
        """the path of the dataset files to be saved"""
        return os.path.join("data", self.name)

    @property
    def n_class(self):
        return self._n_class

    # ----------------------------------------------
    def torch_loader(self, istrain):
        """
            Fetch data by torch.utils.data.Dataset
            Create dataloader
            Input:
                istrain: flag condition for download training or test data
        """

        dataset_class = getattr(torchvision.datasets, self.name)
        dataset = dataset_class(
            root="data",
            train=istrain,
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )
  

        return dataset

    # ----------------------------------------------
