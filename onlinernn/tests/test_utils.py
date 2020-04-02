from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

# ----------------------------------------------

def show_shift(imgs, nrow, result_dir, filename):
    """
    Plot MNIST shift data 
    """

    imgs = vutils.make_grid(imgs, pad_value=1, nrow=nrow)
    # npimgs = imgs.numpy()
    plt.figure(figsize=(4, 6))
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(result_dir + "/" + filename)

