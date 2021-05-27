import torch
import numpy as np

# ----------------------------------------------

def roll(x, n):
    """
    Roll tensor like numpy.roll
    """  

    return torch.cat((x[:, -n:, :], x[:, :-n, :]), dim=1)

def loop_queue(dataset):
    """
    Create new MNIST dataset by shifting each row from top to the bottom until the whole image is exhausted 
    """

    newdata = []
    newtargets = []

    for i in range(len(dataset)):
    # for i in range(56):
     
        idata, itarget = dataset[i]
        nrow = idata.size()[1]

        newdata.append(idata)
        newtargets.append(itarget)
        
        # roll bottom row to top for 27 times
        for irow in range(nrow-1):
 
            idata = roll(idata, 1)
       
            newdata.append(idata)
            newtargets.append(itarget)
    return newdata, newtargets
# ----------------------------------------------

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# ----------------------------------------------
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

# ----------------------------------------------
class ReshapePermuteTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        img = img.permute(1, 0, 2)
        return torch.reshape(img, self.new_size)

# ----------------------------------------------

class ExtendAddNoise(object):
    def __init__(self,  noise_index, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.noise_index = noise_index
        
    def __call__(self, tensor):
        np.random.seed(42)
        # extend data dimension and pad with random noise
        print(tensor.shape)
        # noise = torch.randn((tensor.shape[0], 1000, tensor.shape[2])) * self.std + self.mean
        noise = torch.randn((1000, tensor.shape[1])) * self.std + self.mean

        # noise[:, 0:self.noise_index, :] =  tensor
        noise[0:self.noise_index, :] =  tensor

        return torch.Tensor(noise)
      
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)