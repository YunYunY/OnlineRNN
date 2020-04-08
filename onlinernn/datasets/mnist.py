import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.mnistshift_dataset import MNISTShifDataset


# -------------------------------------------------------
# MNIST data
# -------------------------------------------------------

class MNIST(BaseDataset):
    """
        The MNIST database of handwritten digits has a training set of 60,000 examples,
        and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.
        The data range is [0, 255]
        It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data.
        Reference: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, opt):

        self.opt = opt
        self.shuffle = opt.shuffle
        opt.n_class = 10
        opt.feature_shape = 28
        opt.seq_len = 28
        super(MNIST, self).__init__(opt)
        # ToTensor convert data from 0-255 to 0-1, then normalize with mean and std
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        # one-hot encoding for target while read in data 
        # self.target_transform = transforms.Compose(
        #     [transforms.Lambda(lambda target: torch.eye(self.n_class)[target])]
        # )
        self.target_transform = transforms.Compose([])
        istrain = (opt.istrain or opt.continue_train)
        self.dataset, self.dataloader = self.torch_loader(istrain=istrain)
        print(f"Total datasize is {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

# -------------------------------------------------------
# MNIST shift data
# -------------------------------------------------------

class MNISTShift(MNIST):
    def __init__(self, opt):
        super(MNISTShift, self).__init__(opt)
    # ----------------------------------------------
    def torch_loader(self, istrain):
        """
            Fetch data by torch.utils.data.Dataset
            Create dataloader
            Args:
                istrain: flag condition for getting training or test data
        """

        # dataset_class = getattr(torchvision.datasets, self.name)
        dataset_orig = datasets.MNIST(
            root="data",
            train=istrain,
            download=self.download,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        # use the shift data as training data, original MNIST as testing
        if istrain:
            dataset = MNISTShifDataset(dataset_orig)
        else:
            dataset = dataset_orig
        # print(f"Total datasize is {len(dataset)}")


        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_threads,
            )

        return dataset, dataloader

    # ----------------------------------------------
