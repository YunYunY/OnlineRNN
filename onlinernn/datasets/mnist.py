import torch
from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data_utils import queue_loop 

class MNIST(BaseDataset):
    """
        The MNIST database of handwritten digits has a training set of 60,000 examples,
        and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.
        It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data.
        Reference: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, opt):

        self.opt = opt
        self.shift_data =opt.shift_data
        opt.n_class = 10
        opt.feature_shape = 28
        super(MNIST, self).__init__(opt)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.fliptransform = self.transform
        # one-hot encoding for target while read in data
        self.target_transform = transforms.Compose(
            [transforms.Lambda(lambda target: torch.eye(self.n_class)[target])]
        )
        istrain = (opt.istrain or opt.continue_train)
        self.dataset = self.torch_loader(istrain=istrain)

        # expand each image to 28 new images by shifting each row to the end of the data like a cycle queue
        if self.shift_data:
            self.dataset = queue_loop(self.dataset)

        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_threads,
            )

    def __len__(self):
        return len(self.dataset)