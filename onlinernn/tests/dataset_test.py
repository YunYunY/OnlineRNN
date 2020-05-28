import glob, os
import torch
import numpy as np
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTPadNoise, MNISTPixel, MNISTPixelPermute, MNISTShift, MNISTPermute
from onlinernn.datasets.cifar import CIFARNoise
from onlinernn.datasets.har import HAR_2
from onlinernn.datasets.dsa import DSA_19
from onlinernn.datasets.adding import ADDING
from onlinernn.datasets.cm import CM
from onlinernn.tests.test_utils import show_shift
from onlinernn.datasets.data_utils import loop_queue
from onlinernn.datasets.mnist_byte import MNIST_byte



# ------------------------------------------------------------

opt = TrainOptions().parse()
opt.test_case = "test"
opt.batch_size = 64
opt.istrain = True
# ------------------------------------------------------------

# def test_MNIST_byte():
#     """
#     Test MNIST class
#     """
#     d = MNIST_byte(opt)
#     batch = next(iter(d.dataloader))    
#     assert list(batch[0][:64].shape) == [64, 784, 1]
#     data = batch[0][:64]
#     result_dir = "result/dataset_test/pixelMNIST_byte"
#     os.makedirs(result_dir, exist_ok=True)
#     data = data.view(64, 1, 28, 28)
#     show_shift(data, 8, result_dir, "pixelMNIST_byte.png")

'''
def test_permutepixelmnist():
    """
    Test MNIST class
    """
    opt.download_data = False
    opt.mnist_standardize = "zeromean"
    d = MNISTPixelPermute(opt)
    batch = next(iter(d.dataloader))
    assert list(batch[0][:64].shape) == [64, 784, 1]
    data = batch[0][:64]
 
    result_dir = "result/dataset_test/permutepixelMNIST"
    os.makedirs(result_dir, exist_ok=True)
    data = data.view(64, 1, 28, 28)
    show_shift(data, 8, result_dir, "permutepixelMNIST.png")


def test_pixelmnist():
    """
    Test MNIST class
    """
    opt.download_data = False
    opt.mnist_standardize = "zeromean"
    d = MNISTPixel(opt)
    batch = next(iter(d.dataloader))
    assert list(batch[0][:64].shape) == [64, 784, 1]
    data = batch[0][:64]
    result_dir = "result/dataset_test/pixelMNIST"
    os.makedirs(result_dir, exist_ok=True)
    data = data.view(64, 1, 28, 28)
    show_shift(data, 8, result_dir, "pixelMNIST.png")


def test_dsa19():
    d = DSA_19(opt)
    assert len(d) == 4560
    batch = next(iter(d.dataloader))
    assert list(batch[0][:64].shape) == [64, 125, 45]
    assert list(batch[1][:64].shape) == [64, 1]


def test_har2():
    d = HAR_2(opt)
    if opt.istrain:
        assert len(d) == 7352
    else:
        assert len(d) == 2947
    batch = next(iter(d.dataloader))
    print(batch[0][:64].mean())
    assert list(batch[0][:64].shape) == [64, 128, 9]
    assert list(batch[1][:64].shape) == [64, 1]




def test_adding():
    opt.N_TRAIN = 500
    opt.N_TEST = 100
    opt.seq_len = 30
    d = ADDING(opt)
    
    batch = next(iter(d.dataloader))
    print(batch[1][:64].shape)


def test_copymemory():
    opt.N_TRAIN = 500
    opt.N_TEST = 100
    opt.seq_len = 30
    d = CopyMemory(opt)
    
    batch = next(iter(d.dataloader))
    print(batch[1][:64].shape)
'''
def test_cifar():
    opt.istrain = True
    opt.batch_size = 512

    d = CIFARNoise(opt)
    batch = next(iter(d.dataloader))
    # print(d.dataset.data.min())
    print(batch[0][:][:, :, :, 0].mean())
    # print(batch[0][:][:, :, :, 1].std())
    # print(batch[0][:][:, :, :, 2].std())
    # for i in range(3):
    #     print(d.dataset.data[:, :, :, i].mean()/255)
    #     print(d.dataset.data[:, :, :, i].std()/255)

    data = batch[0][:64, :, 0:40, :]
    result_dir = "result/dataset_test/CIFARNoise"
    os.makedirs(result_dir, exist_ok=True)
    # visually check image after shifting
    show_shift(data, 8, result_dir, "CIFARNoise.png")





def test_mnistpadnoise():
    d = MNISTPadNoise(opt)
    batch = next(iter(d.dataloader))
    print(batch[0][:64].mean())
    print(batch[0][:64].std())
    data = batch[0][:64, :, 0:30, :]
    print(data.shape)
    result_dir = "result/dataset_test/MNISTNoise"
    os.makedirs(result_dir, exist_ok=True)
    # visually check image after shifting
    show_shift(data, 8, result_dir, "MNISTNoise.png")

'''

def test_mnist():
    """
    Test MNIST class
    """
    opt.download_data = False
    d = MNIST(opt)
    assert opt.feature_shape == 28
    assert opt.n_class == 10
    assert os.path.exists("data/MNIST/")
    assert os.path.exists("data/MNIST/processed/")
    assert os.path.exists("data/MNIST/raw/")
    assert os.path.isfile("data/MNIST/processed/training.pt")
    assert os.path.isfile("data/MNIST/processed/test.pt")
    

    # calculate mean and std of the whole dataset https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/5

    # print(d.dataset.data.float().mean()/255)
    # print(d.dataset.data.float().std()/255)

    batch = next(iter(d.dataloader))
    
    assert list(batch[0][:64].shape) == [64, 1, 28, 28]
    assert list(batch[1][:64].shape) == [64]
    # test loop_queue 
    data_shift = loop_queue(d.dataset)
    print(batch[1][:64])


# ------------------------------------------------------------

def test_mnistshift():
    """
    Test MNISTShift class
    """
    opt.batch_size = 56
    opt.shuffle = False
    d = MNISTShift(opt)

    # assert len(d.dataset) == 10000
    batch = next(iter(d.dataloader))
    data = batch[0][:opt.batch_size]
    # print(data.min())
    # print(data.max())
    result_dir = "result/dataset_test/MNISTShift"
    os.makedirs(result_dir, exist_ok=True)
    # visually check image after shifting
    show_shift(data, 7, result_dir, "MNISTShift.png")

# ------------------------------------------------------------

def test_mnistpermute():
    """
    Test MNISTPermute class
    """
    print(opt.continue_train)
    opt.batch_size = 64
    d = MNISTPermute(opt)
    assert len(d.dataset) == 10000
    batch = next(iter(d.dataloader))
    data = batch[0][:opt.batch_size]
    result_dir = "result/dataset_test/MNISTPermute"
    os.makedirs(result_dir, exist_ok=True)
    # visually check image after shifting
    show_shift(data, 8, result_dir, "MNISTPermute.png")

'''