import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import torchvision
from onlinernn.datasets.base_dataset import BaseDataset
from onlinernn.datasets.mnistshift_dataset import MNISTShifDataset
from onlinernn.datasets.data_utils import ExtendAddNoise
# -------------------------------------------------------
# Noisy CIFAR
# 50000 training, 10000 testing, [3, 32, 32]
# -------------------------------------------------------
class CIFARNoise(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.shuffle = opt.shuffle
        opt.n_class = 10
        opt.feature_shape = 32
        opt.seq_len = 1000
        super(CIFARNoise, self).__init__(opt)
        # ToTensor convert data from 0-255 to 0-1
        data_transforms = [transforms.ToTensor()]
        # Normalize to 0 mean and unit std
        data_transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
        data_transforms.append(ExtendAddNoise(opt.feature_shape, 0., 1.))
        # mnist_transforms.append(ReshapeTransform((784, 1)))
        self.transform = transforms.Compose(data_transforms)

        self.target_transform = transforms.Compose([])
        istrain = (opt.istrain or opt.continue_train)
        self.dataset, self.dataloader = self.torch_loader(istrain=istrain)
        print(f"Total datasize is {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)


