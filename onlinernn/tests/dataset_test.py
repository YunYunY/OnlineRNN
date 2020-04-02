from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTShift
import glob, os
import torch
from onlinernn.tests.test_utils import show_shift
from onlinernn.datasets.data_utils import loop_queue


# ------------------------------------------------------------

opt = TrainOptions().parse()
opt.test_case = "test"
opt.batch_size = 64

# ------------------------------------------------------------

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
    batch = next(iter(d.dataloader))
    assert list(batch[0][:64].shape) == [64, 1, 28, 28]
    assert list(batch[1][:64].shape) == [64, 10]
    # test loop_queue 
    data_shift = loop_queue(d.dataset)



# ------------------------------------------------------------

def test_mnistshift():
    """
    Test MNISTShift class
    """
    opt.batch_size = 56
    opt.shuffle = False
    d = MNISTShift(opt)
    assert len(d.dataset) == 280000
    batch = next(iter(d.dataloader))
    # assert list(batch[0][:opt.batch_size].shape) == [opt.batch_size, 1, 28, 28]
    # assert list(batch[1][:opt.batch_size].shape) == [opt.batch_size, 10]
    data = batch[0][:opt.batch_size]
    
    result_dir = "result/dataset_test/MNISTShift"
    os.makedirs(result_dir, exist_ok=True)

    # visually check image after shifting
    show_shift(data, 7, result_dir, "MNISTShift.png")

