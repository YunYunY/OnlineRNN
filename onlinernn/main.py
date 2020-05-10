import torch
import numpy as np
from exp.expConfig import ExpConfig
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTPixel, MNISTShift
from onlinernn.datasets.mnist_byte import MNIST_byte
from onlinernn.datasets.har import HAR_2
from onlinernn.datasets.dsa import DSA_19
from onlinernn.models.setting import Setting
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.rnn_stopbp import StopBPRNN
from onlinernn.models.rnn_tbptt import TBPTT
from onlinernn.models.rnn_irnn import IRNN
from onlinernn.models.rnn_ind import IndRNN
from onlinernn.models.setting import RNN

torch.manual_seed(42)
np.random.seed(42)
torch.set_printoptions(precision=8)
"""
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
"""
# -----------------------------------------------------------------------------------------------
# Get training options
# -----------------------------------------------------------------------------------------------
opt = TrainOptions().parse()
# Hardcode some parameters for test
if not opt.istrain:
    opt.num_threads = 0  # test code only supports num_threads = 1
    # opt.num_test = 1  # how many test batches to run
# -----------------------------------------------------------------------------------------------

if opt.taskid == 0:
    for T in opt.T:
        print(f"----------------- Truncation T is {T} -----------------")
        opt.T_ = T

        d = MNIST(opt)
        s = RNN(opt)
        m = IRNN(opt)
        p = ExpConfig(dataset=d, setting=s, model=m)
        p.run()
# -----------------------------------------------------------------------------------------------
# Vanishing Gradient Solver FGSM
# -----------------------------------------------------------------------------------------------

if opt.taskid == 1:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    d = HAR_2(opt)
    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 2:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.hidden_size = 64
    opt.batch_size = 128
    opt.predic_task = 'Softmax'
    d = DSA_19(opt)
    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = DSA_19(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
# Vanishing Gradient Solver FGSM + Adam with TBPTT
# -----------------------------------------------------------------------------------------------

if opt.taskid == 3:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.subsequene = True
    opt.subseq_size = 128
    opt.niter = 49
    opt.add_noise = True
    opt.iterT = 4
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 4:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = 'Adam'
    opt.subsequene = True
    opt.subseq_size = 128
    opt.niter = 49
    opt.add_noise = True
    opt.iterT = 4

    d = HAR_2(opt)
    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------
# Repeat result of sequential MNIST publish on indRNN
# Reference: https://github.com/Sunnydreamrain/IndRNN_pytorch/tree/master/pixelMNIST
# -----------------------------------------------------------------------------------------------
if opt.taskid == 5:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.niter_decay = 320
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------
if opt.taskid == 6:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.niter_decay = 320
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 7:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 4
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.niter_decay = 320
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()