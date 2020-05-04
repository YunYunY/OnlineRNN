import torch
import torch.nn as nn
import numpy as np
from os import path
import functools 
from torchsummary import summary
from torch.autograd import Variable
from onlinernn.models.fgsm import FGSM, MultipleOptimizer
from onlinernn.models.adam import Adam 
from onlinernn.models.setting import Setting, RNN
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.rnn_stopbp import StopBPRNN
from onlinernn.models.rnn_tbptt import TBPTT
from onlinernn.models.rnn_irnn import IRNN
from onlinernn.datasets.mnist import MNIST, MNISTShift
from onlinernn.options.train_options import TrainOptions
from onlinernn.models.networks import SimpleRNN
from onlinernn.exp.expConfig import ExpConfig


opt = TrainOptions().parse()
opt.istrain = True
opt.device = torch.device(
    "cuda" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
)
opt.T_ = 4

device = torch.device("cuda" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
# -----------------------------------------------------
# FGSM Optimizer
# -----------------------------------------------------
class simplelinear(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(simplelinear, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

def test_fgsm():

    # create dummy data for training
    x_values = [i for i in range(2)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    inputDim = 1        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    lr = 0.01 
    epochs = 10

    model = simplelinear(inputDim, outputDim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss() 
    optimizer = FGSM(model.parameters(), lr=lr, iterT=opt.T_)
    # optimizer =  MultipleOptimizer(FGSM(model.parameters(), lr=lr, iterT=opt.T_, mergeadam=True), 
                        # Adam(model.parameters(), lr=lr))
    # optimizer =  torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for p in model.parameters():
        p.data.fill_(0)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        optimizer.zero_grad()
            
        # get output from the model, given the inputs
        outputs = model(inputs)
        print(f'----epoch {epoch}-----------------')
        
        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # print(loss)
        
        # get gradients w.r.t to parameters
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(param.data)
        #         print(param.grad)
        # update parameters
        # optimizer.step()
        optimizer.step(epoch+1, True)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(param.data)
        #         print(param.buf)
        #         print(param.grad)
        print('-----------------------')

        # print('epoch {}, loss {}'.format(epoch, loss.item()))


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
# opt.niter = 9
# opt.niter_decay = 0
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
    

# opt.niter = 9
# opt.niter_decay = 0
# def test_StopBPRNN():
#     d = MNIST(opt)
#     s = RNN(opt)
#     m = StopBPRNN(opt)
#     p = ExpConfig(dataset=d, setting=s, model=m)
#     s.setup(dataset=d, model=m)
#     p.run()

# opt.niter = 0
# opt.niter_decay = 0
# def test_TBPTT():
#     d = MNIST(opt)
#     s = RNN(opt)
#     m = TBPTT(opt)
#     p = ExpConfig(dataset=d, setting=s, model=m)
#     s.setup(dataset=d, model=m)
#     p.run()


# def test_IRNN():
#     d = MNIST(opt)
#     s = RNN(opt)
#     m = IRNN(opt)
#     p = ExpConfig(dataset=d, setting=s, model=m)
#     batch = next(iter(d.dataloader))
#     s.setup(dataset=d, model=m)
#     m.data = batch
#     m.init_net()
#     m.set_input()
#     m.init_states()
#     y = m.rnn_model(m.inputs, m.states)
#     print(y.shape)
#     print(m.labels.shape)
