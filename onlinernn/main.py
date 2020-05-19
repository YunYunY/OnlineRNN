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
# FGSM grad/2norm of grad, sum every sign, final update with Adam(Deltaw), 1/t coefficient, iterT=1, lr=2e-4

if opt.taskid == 1:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
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



# -----------------------------------------------------------------------------------------------
# FGSM grad/2norm of grad, sum every sign, final update with Adam(Deltaw), 1/t coefficient, iterT=1, lr=2e-4

if opt.taskid == 2:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_RMSProp"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
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

# -----------------------------------------------------------------------------------------------

if opt.taskid == 8:
    opt.optimizer = "Adam" 
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
if opt.taskid == 9:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
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
if opt.taskid == 10:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 20
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
# average of sign
if opt.taskid == 11:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 20
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
# No FGSM sign, sum every sign, final update with average of sum,
if opt.taskid == 12:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 30
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
# No FGSM sign, sum every sign, final update with average of sum,
if opt.taskid == 13:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 40
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
# No FGSM sign, sum every sign, final update with average of sum,
if opt.taskid == 14:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 50
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
# FGSM sign, sum every sign, final update with average of sum, 1/t coefficient, iterT=40, lr=2e-4
if opt.taskid == 15:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 40
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
# FGSM sign, sum every sign, final update with average of sum, 1/t coefficient, iterT=40, lr=2e-3
if opt.taskid == 16:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-3
    opt.end_rate = 1e-5
    opt.iterT = 40
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
# FGSM sign, sum every sign, final update with average of sum, 2/(t+2) coefficient, iterT=40, lr=2e-4
if opt.taskid == 17:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 40
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
# FGSM sign, sum every sign, final update with average of sum, 2/(t+2) coefficient, iterT=40, lr=2e-3
if opt.taskid == 18:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-3
    opt.end_rate = 1e-5
    opt.iterT = 40
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
# FGSM sign, sum every sign, final update with average of sum, 1/t coefficient, iterT=40, lr=2e-4
if opt.taskid == 19:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-5
    opt.end_rate = 1e-7
    opt.iterT = 40
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
# FGSM grad/2norm of grad, sum every sign, final update with average of sum, 1/t coefficient, iterT=40, lr=2e-4
if opt.taskid == 20:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 40
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
# FGSM grad/2norm of grad, sum every sign, final update with average of sum, 1/t coefficient, iterT=1, lr=2e-4
if opt.taskid == 21:
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
# tbptt with indRNN no BN layer
if opt.taskid == 22:
    opt.optimizer = "FGSM_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 4
    opt.subsequene = True
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
# pixelMNIST with tbptt

if opt.taskid == 23:

    opt.optimizer = "Adam"
    opt.subsequene = True
    opt.mnist_standardize = "originalmean"
    opt.subseq_size = 28
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = True
    opt.niter = 10000000-1
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
# tbptt with indRNN no BN layer, 1 layer 
if opt.taskid == 24:
    opt.optimizer = "FGSM_Adam" 
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 4
    opt.subsequene = True
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
# pixelMNIST with tbptt

if opt.taskid == 25:

    opt.optimizer = "Adam"
    opt.subsequene = True
    opt.mnist_standardize = "originalmean"
    opt.subseq_size = 196
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = False
    opt.niter = 759
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
# pixelMNIST with tbptt

if opt.taskid == 26:

    opt.optimizer = "Adam"
    opt.subsequene = True
    opt.mnist_standardize = "originalmean"
    opt.subseq_size = 28
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = False
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
# pixelMNIST with tbptt

if opt.taskid == 27:

    opt.optimizer = "Adam"
    opt.subsequene = True
    opt.mnist_standardize = "originalmean"
    opt.subseq_size = 196
    opt.hidden_size = 128
    opt.niter_decay = 200
    opt.endless_train = False
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
# tbptt with indRNN no BN layer
if opt.taskid == 28:
    opt.optimizer = "Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 4
    opt.subsequene = True
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


# tbptt with indRNN no BN layer same as 22
if opt.taskid == 29:
    opt.optimizer = "FGSM_Adam" 
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 4
    opt.subsequene = True
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

