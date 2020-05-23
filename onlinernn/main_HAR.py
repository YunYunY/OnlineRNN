import torch
import numpy as np
from exp.expConfig import ExpConfig
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTPixel, MNISTPixelPermute, MNISTShift
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
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
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

if opt.taskid == 1:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    d = HAR_2(opt)
    opt.subsequene = True
    opt.subseq_size = 16

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

if opt.taskid == 2:
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

if opt.taskid == 3:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 30
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

if opt.taskid == 4:
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

if opt.taskid == 5:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 30
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

if opt.taskid == 6:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    d = HAR_2(opt)
    opt.subsequene = True
    opt.subseq_size = 16

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

if opt.taskid == 20:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.add_noise = True
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

if opt.taskid == 21:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

    d = HAR_2(opt)
    opt.subsequene = True
    opt.subseq_size = 16

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

if opt.taskid == 22:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

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

if opt.taskid == 23:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 30
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

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

if opt.taskid == 24:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

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

if opt.taskid == 25:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 30
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

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

if opt.taskid == 26:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.niter = 199
    opt.add_noise = True

    d = HAR_2(opt)
    opt.subsequene = True
    opt.subseq_size = 16

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


