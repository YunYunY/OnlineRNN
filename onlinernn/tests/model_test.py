import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from os import path
from onlinernn.models.setting import Setting, RNN
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.rnn_stopbp import SoptBPRNN
from onlinernn.datasets.mnist import MNIST, MNISTShift
from onlinernn.options.train_options import TrainOptions
from onlinernn.models.networks import SimpleRNN
from onlinernn.exp.expConfig import ExpConfig


opt = TrainOptions().parse()
opt.istrain = True
opt.device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
)

# -----------------------------------------------------
# VanillaRNN
# -----------------------------------------------------
opt.feature_shape = 28
opt.n_class = 10
# def test_set_input():
#     """
#         Inut and label size feed into RNN
#     """
#     d = MNIST(opt)
#     m = VanillaRNN(opt)
#     m.data = next(iter(d.dataloader))
#     m.set_input()
#     assert list(m.inputs.shape) == [64, 28, 28]
#     assert list(m.labels.shape) == [64]

# accuracy should be around 95%
opt.niter = 9
opt.niter_decay = 0
# def test_VanillaRNN():
#     d = MNIST(opt)
#     s = RNN(opt)
#     m = VanillaRNN(opt)
#     p = ExpConfig(dataset=d, setting=s, model=m)
#     s.setup(dataset=d, model=m)
#     p.run()

# -----------------------------------------------------
# StopBPRNN
# -----------------------------------------------------
# def test_StopBPRNN_train():
#     d = MNIST(opt)
#     m = SoptBPRNN(opt)
#     m.data = next(iter(d.dataloader))
#     m.init_net()
#     m.init_loss()
#     m.init_optimizer()
#     m.set_input()
#     m.set_output()
#     m.train()
    

opt.niter = 9
opt.niter_decay = 0
def test_StopBPRNN():
    d = MNISTShift(opt)
    s = RNN(opt)
    m = SoptBPRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m)
    s.setup(dataset=d, model=m)
    p.run()
