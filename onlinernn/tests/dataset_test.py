import glob, os
import torch
import numpy as np
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTShift, MNISTPermute
from onlinernn.datasets.har import HAR_2
from onlinernn.tests.test_utils import show_shift
from onlinernn.datasets.data_utils import loop_queue


# ------------------------------------------------------------

opt = TrainOptions().parse()
opt.test_case = "test"
opt.batch_size = 64
opt.istrain = True
# ------------------------------------------------------------

def test_har2():
    d = HAR_2(opt)
    if opt.istrain:
        assert len(d) == 7352
    else:
        assert len(d) == 2947
    batch = next(iter(d.dataloader))
    assert list(batch[0][:64].shape) == [64, 1152, 1]
    assert list(batch[1][:64].shape) == [64, 1]

'''
def test_mnist():
    """
    Test MNIST class
    """
    opt.download_daa = False
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