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
opt.iterT = 10

if opt.taskid == 100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 1e-3
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


if opt.taskid == 101:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
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


if opt.taskid == 102:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
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


if opt.taskid == 103:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 1e-3
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


if opt.taskid == 104:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 1e-3
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

if opt.taskid == 200:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
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


if opt.taskid == 201:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-4
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


if opt.taskid == 202:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-4
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


if opt.taskid == 203:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-4
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


if opt.taskid == 204:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-4
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

if opt.taskid == 300:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 6e-4
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


if opt.taskid == 301:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 6e-4
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


if opt.taskid == 302:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 6e-4
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


if opt.taskid == 303:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 6e-4
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


if opt.taskid == 304:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 6e-4
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

if opt.taskid == 400:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-5
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


if opt.taskid == 401:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-5
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


if opt.taskid == 402:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-5
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


if opt.taskid == 403:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-5
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


if opt.taskid == 404:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-5
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

if opt.taskid == 500:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-2
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


if opt.taskid == 501:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-2
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


if opt.taskid == 502:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-2
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


if opt.taskid == 503:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-2
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


if opt.taskid == 504:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-2
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

if opt.taskid == 600:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 1e-3
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


if opt.taskid == 601:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
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


if opt.taskid == 602:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
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


if opt.taskid == 603:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 1e-3
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


if opt.taskid == 604:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 1e-3
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

if opt.taskid == 700:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
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


if opt.taskid == 701:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-4
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


if opt.taskid == 702:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-4
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


if opt.taskid == 703:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-4
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


if opt.taskid == 704:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-4
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

if opt.taskid == 800:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 6e-4
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


if opt.taskid == 801:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 6e-4
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


if opt.taskid == 802:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 6e-4
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


if opt.taskid == 803:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 6e-4
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


if opt.taskid == 804:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 6e-4
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

if opt.taskid == 900:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-5
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


if opt.taskid == 901:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-5
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


if opt.taskid == 902:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-5
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


if opt.taskid == 903:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-5
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


if opt.taskid == 904:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-5
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

if opt.taskid == 1000:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-2
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


if opt.taskid == 1001:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-2
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


if opt.taskid == 1002:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-2
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


if opt.taskid == 1003:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 2e-2
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


if opt.taskid == 1004:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 512
    opt.lr = 2e-2
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