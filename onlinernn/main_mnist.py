
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

opt.verbose = True

# -----------------------------------------------------------------------------------------------

if opt.taskid == 0:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 15:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




# -----------------------------------------------------------------------------------------------

if opt.taskid == 14:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 2021:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 64
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 2022:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 64
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 2023:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 2014:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 5
    opt.batch_size = 64
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 2015:

    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 5
    opt.batch_size = 64
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
# -----------------------------------------------------------------------------------------------

if opt.taskid == 215:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 64
    opt.lr = 2e-4
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
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
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------
if opt.taskid == 2:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
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


# -----------------------------------------------------------------------------------------------
if opt.taskid == 3:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5
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
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.9
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
if opt.taskid == 5:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.lrgamma = 0.5
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
if opt.taskid == 202:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
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
if opt.taskid == 203:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5

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
if opt.taskid == 204:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.9

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
if opt.taskid == 205:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.lrgamma = 0.5

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
if opt.taskid == 302:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
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
if opt.taskid == 402:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
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
if opt.taskid == 1002:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
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
if opt.taskid == 1202:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
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
if opt.taskid == 1302:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
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
if opt.taskid == 1402:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
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

if opt.taskid == 10:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 610:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 128
    opt.endless_train = False
    opt.niter = 200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 611:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 128
    opt.endless_train = False
    opt.lr = 1e-8
    opt.niter = 200
    # opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 612:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 128
    opt.endless_train = False
    opt.lr = 1e-3
    opt.niter = 400
    opt.niter_decay = 1
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.9
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 613:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 128
    opt.endless_train = False
    opt.lr = 1e-3
    opt.niter = 400
    opt.niter_decay = 1
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.9
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------

if opt.taskid == 110:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.lr = 1e-3
    opt.batch_size = 128
    opt.endless_train = False
    opt.niter = 10000-1
    opt.niter_decay = 1
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.9
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------

if opt.taskid == 11:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 1011:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.batch_size = 512
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.5
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------

if opt.taskid == 111:

    opt.optimizer = "Adam"
    opt.hidden_size = 128
    opt.epoch_count = 0
    # opt.epoch = "2"
    opt.iterT = 1
    opt.lr = 1e-3
    opt.batch_size = 128
    opt.endless_train = False
    opt.niter = 10000-1
    opt.niter_decay = 1
    opt.predic_task = "Softmax"
    opt.mnist_standardize = "zeromean"
    opt.lrgamma = 0.9
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt) 
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 12:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 212:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 312:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 412:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 2412:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------
if opt.taskid == 1002:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 1212:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 1213:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 1214:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 1215:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-100
    opt.subsequene = True
    opt.subseq_size = 196
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 1217:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-100
    opt.subsequene = True
    opt.subseq_size = 196
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 1216:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.subsequene = True
    opt.subseq_size = 196
    opt.niter_decay = 1
    opt.lrgamma = 0.9
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 1312:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
if opt.taskid == 1412:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
if opt.taskid == 3412:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
# -----------------------------------------------------------------------------------------------


if opt.taskid == 13:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 430:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5

    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 431:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.9

    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 432:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    opt.lrgamma = 0.5

    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
# -----------------------------------------------------------------------------------------------


if opt.taskid == 113:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------


if opt.taskid == 1130:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------


if opt.taskid == 119:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.lr = 6e-4
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------


if opt.taskid == 1190:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.lr = 6e-4
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------


if opt.taskid == 1191:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 100
    opt.lr = 6e-4
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------


if opt.taskid == 219:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.lr = 1e-2
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------


if opt.taskid == 2190:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.lr = 1e-2
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
# -----------------------------------------------------------------------------------------------


if opt.taskid == 116:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-200
    opt.niter_decay = 200
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------


if opt.taskid == 114:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    opt.niter_decay = 1
    opt.lrgamma = 0.9
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------


if opt.taskid == 117:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.niter_decay = 1
    opt.lrgamma = 0.9
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

# -----------------------------------------------------------------------------------------------


if opt.taskid == 115:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 512
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 118:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 301:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.lr = 6e-4
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 1301:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.lr = 6e-4
    opt.lrgamma = 0.9
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    opt.niter_decay = 1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 2301:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.iterT = 1
    opt.lr = 6e-4
    opt.lrgamma = 0.5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 400:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.lr = 1e-3
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 433:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.lr = 1e-3
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 1433:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.lr = 1e-3
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1-100
    opt.niter_decay = 100
    opt.lrgamma = 0.5
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 1400:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.lr = 1e-3
    opt.lrgamma = 0.9
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1
    opt.niter_decay = 1
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 2400:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.iterT = 5
    opt.lr = 1e-3
    opt.lrgamma = 0.5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000-1-200
    opt.niter_decay = 200
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
# Repeat result of sequential MNIST publish on indRNN
# Reference: https://github.com/Sunnydreamrain/IndRNN_pytorch/tree/master/pixelMNIST
# -----------------------------------------------------------------------------------------------

if opt.taskid == 20:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
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

if opt.taskid == 21:
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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


if opt.taskid == 721:
    opt.optimizer = "FGSM_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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



if opt.taskid == 621:
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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



if opt.taskid == 623:
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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


if opt.taskid == 22:
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



if opt.taskid == 22:
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

if opt.taskid == 23:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
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


if opt.taskid == 24:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
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



if opt.taskid == 2500:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 399
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




if opt.taskid == 625:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    input_size = 1
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
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




if opt.taskid == 26:
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


if opt.taskid == 2700:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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




if opt.taskid == 627:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
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


if opt.taskid == 2800:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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


if opt.taskid == 628:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
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


if opt.taskid == 2900:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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



if opt.taskid == 622:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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




if opt.taskid == 624:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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


if opt.taskid == 629:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
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





if opt.taskid == 3000:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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



if opt.taskid == 630:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 10000000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
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


if opt.taskid == 3100:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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




if opt.taskid == 3200:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 100
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



if opt.taskid == 32500:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 100
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 517:
    opt.optimizer = "irnn_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 300-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()





if opt.taskid == 32600:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 100
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 512:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 327:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 6e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000000-1
    opt.niter_decay = 640
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 32800:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 100
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 513:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 329:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 64
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 5
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000000-1
    opt.niter_decay = 640
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 41000:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 100
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 514:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.constrain_grad = False
    opt.constrain_U = False
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 411:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 10
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000000-1
    opt.niter_decay = 640
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 51000:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 100
    opt.niter_decay = 320
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 515:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.constrain_grad = False
    opt.constrain_U = False
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 516:
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 716:
    opt.optimizer = "FGSM_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 7000:
    opt.optimizer = "FGSM_Adam" #"FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 2e-4
    opt.iterT = 1
    opt.mnist_standardize = "zeromean"
    opt.endless_train = True
    opt.niter = 1000-1
    opt.niter_decay = 320
    opt.subsequene = True
    opt.subseq_size = 196
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 511:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 128
    opt.batch_size = 32
    opt.predic_task = "Softmax"
    opt.lr = 1e-3
    opt.iterT = 30
    opt.mnist_standardize = "zeromean"
    opt.endless_train = False
    opt.niter = 10000000-1
    opt.niter_decay = 640
    d = MNISTPixelPermute(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = MNISTPixelPermute(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()












