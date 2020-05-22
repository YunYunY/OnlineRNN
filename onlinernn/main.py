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


# -----------------------------------------------------------------------------------------------
# Vanishing Gradient Solver FGSM
# -----------------------------------------------------------------------------------------------
# FGSM grad/2norm of grad, final update with Adam(Deltaw), 1/t coefficient, iterT=1, lr=2e-4

# -----------------------------------------------------------------------------------------------

if opt.taskid == 0:

    opt.optimizer = "Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 1:

    opt.optimizer = "FGSM_Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------

if opt.taskid == 2:

    opt.optimizer = "FGSM_Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 3:

    opt.optimizer = "FGSM_Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 30
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 4:

    opt.optimizer = "Adam"
    opt.mnist_standardize = "zeromean"
    opt.subsequene = True
    opt.subseq_size = 196
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 5:

    opt.optimizer = "Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 6:

    opt.optimizer = "FGSM_Adam"
    opt.mnist_standardize = "zeromean"
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.predic_task = "Softmax"
    opt.lrgamma = 0.5
    opt.iterT = 1
    opt.batch_size = 512
    d = MNISTPixel(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixel(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------
if opt.taskid == 100:
    opt.optimizer = "Adam" 
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
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
if opt.taskid == 101:
    opt.optimizer = "Adam" 
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.niter_decay = 320
    opt.subsequene = True
    opt.subseq_size = 196
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
if opt.taskid == 102:
    opt.optimizer = "FGSM_Adam" 
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
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