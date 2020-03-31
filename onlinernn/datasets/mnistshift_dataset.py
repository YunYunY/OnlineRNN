from torch.utils.data import Dataset
from PIL import Image
from onlinernn.datasets.data_utils import loop_queue

# -------------------------------------------------------
# Custom MNIST dataset
# Expand each image to 28 new images by shifting each row to the end of the data like a cycle queue
# -------------------------------------------------------

class MNISTShifDataset(Dataset):
    
    """    
        Create custom MNIST dataset by shifting each row from top to the bottom until the whole image is exhausted 
    """

    def __init__(self, mnistdataset, transform=None, target_transform=None):
        """
        Args:
            mnistdataset: the original mnist dataset
        """
        self.data, self.targets = loop_queue(mnistdataset)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')
        # img = Image.fromarray(img, mode='L')

        return img, target

