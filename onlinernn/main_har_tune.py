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
from onlinernn.models.rnn_irnn import IRNN
from onlinernn.models.setting import RNN

# torch.manual_seed(42)
# np.random.seed(42)
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
opt.iterT = 1
opt.log = True
opt.LSTM = False
opt.add_noise = False
opt.constrain_grad = False
opt.constrain_U = False
opt.verbose = True
opt.test_batch = True


if opt.taskid == 100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'GD'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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


if opt.taskid == 103011:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.ratio = 0.01
    opt.sparse = False
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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



if opt.taskid == 10333:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'GD'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 199
    opt.niter_decay = 0
    opt.lrgamma = 0.1
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

if opt.taskid == 10100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = False
    opt.meta = 1
    opt.task = 'GD'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
    # opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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

if opt.taskid == 102:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = False
    opt.meta = 1
    opt.task = 'GD'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    # opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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



if opt.taskid == 200:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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


if opt.taskid == 20333:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 199
    opt.niter_decay = 0
    opt.lrgamma = 0.1
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



if opt.taskid == 2018:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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



if opt.taskid == 202:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = False
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    # opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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



if opt.taskid == 300:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'NAG'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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


if opt.taskid == 3017:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = False
    opt.meta = 1
    opt.task = 'NAG'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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




if opt.taskid == 30333:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = True
    opt.meta = 1
    opt.task = 'NAG'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    # opt.lr = 1e-3
    opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 199
    opt.niter_decay = 0
    opt.lrgamma = 0.1
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


if opt.taskid == 302:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.sparse = False
    opt.meta = 1
    opt.task = 'NAG'
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    # opt.lr = 1e-2
    opt.endless_train = False
    opt.niter = 99
    opt.niter_decay = 200
    opt.lrgamma = 0.1
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



