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




