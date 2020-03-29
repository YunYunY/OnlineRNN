from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST
import glob, os
import joblib
import torch


# ------------------------------------------------------------
opt = TrainOptions().parse()
opt.test_case = "test"


def test_mnist():
    d = MNIST(opt)
    assert opt.feature_shape == 28
    assert opt.n_class == 10
    # assert os.path.exists("data/MNIST/")
    # assert os.path.exists("data/MNIST/processed/")
    # assert os.path.exists("data/MNIST/raw/")
    # assert os.path.isfile("data/MNIST/processed/training.pt")
    # assert os.path.isfile("data/MNIST/processed/test.pt")
    # batch = next(iter(d.dataloader))

    # assert list(batch[0][:64].shape) == [64, 1, 28, 28]
    # assert list(batch[1][:64].shape) == [64, 10]
    # data = batch[0][:64]
    # result_dir = "result/dataset_test/MNIST"
    # os.makedirs(result_dir, exist_ok=True)
    # imshow(torch.reshape(data, (64, 1, 28, 28)),
    #     result_dir,
    #     8, "MNISTOriginal.png")
    # flipdata = canny(data, use_cuda = False)
    # imshow(torch.reshape(flipdata, (64, 1, 28, 28)),
    #     result_dir,
    #     8, "MNISTEdge.png")
    # data[:, 0, 14, :] = 1-data[:, 0, 14, :]
    # flipdata = data
    # # flipdata = canny(data, use_cuda = False)

    # imshow(torch.reshape(flipdata, (64, 1, 28, 28)),
    #     result_dir,
    #     8, "MNISTFlip.png")
