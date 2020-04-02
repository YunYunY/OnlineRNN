import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from os import path
from onlinernn.models.setting import Setting, RNN
from onlinernn.models.rnn_vanilla import VanillaRNN
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
opt.n_layers, opt.batch_size, opt.hidden_size = 1, 64, 512
def test_init_hidden():
    d = MNIST(opt)
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m)
    s.setup(dataset=d, model=m)
    m.init_hidden()
    assert list(m.state.shape) == [1, 64, 512]

opt.niter = 1
opt.niter_decay = 0
# -----------------------------------------------------
def test_VanillaRNN():
    d = MNIST(opt)
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m)
    s.setup(dataset=d, model=m)
    m.init_net()
    print(m.rnn_model)
    p.run()