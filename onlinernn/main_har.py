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
opt.verbose = True
# opt.LSTM = True
opt.rad = 1e-3
opt.niter_decay = 200
opt.niter = 99

# opt.lrgamma = 0.5

if opt.taskid == 1000:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
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


if opt.taskid == 0:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
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



if opt.taskid == 900:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.clip_grad = True
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
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

if opt.taskid == 202:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 302:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 402:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 502:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 902:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.weight_decay = 0.1
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 903:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.weight_decay = 0.01
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 904:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.weight_decay = 1
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 905:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.weight_decay = 1e-3
    opt.iterT = 1

    opt.endless_train = False
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

if opt.taskid == 906:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.weight_decay = 1e-4
    opt.iterT = 1

    opt.endless_train = False
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
    opt.iterT = 5
    opt.endless_train = False
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
    opt.batch_size = 32
    opt.lr = 6e-4
    opt.rad = opt.lr
    opt.iterT = 10
    opt.endless_train = False
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
    opt.iterT = 1
    opt.endless_train = False
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

if opt.taskid == 6:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 6e-4
    opt.iterT = 10
    opt.endless_train = False
    opt.add_noise = False

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

if opt.taskid == 7:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 7352
    opt.lr = 6e-4
    opt.rad = opt.lr
    opt.iterT = 10
    opt.endless_train = False
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

if opt.taskid == 8:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
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


if opt.taskid == 9:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 6e-4
    opt.iterT = 5
    opt.endless_train = False
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
    
if opt.taskid == 10:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
    opt.iterT = 10
    opt.endless_train = False
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
    
if opt.taskid == 11:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-2
    opt.iterT = 1
    opt.LSTM = True
    opt.endless_train = False
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
    
if opt.taskid == 12:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-2
    opt.iterT = 5
    opt.LSTM = True
    opt.endless_train = False
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
    
if opt.taskid == 13:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-2
    opt.iterT = 10
    opt.LSTM = True
    opt.endless_train = False
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
    
if opt.taskid == 14:
    opt.optimizer = "irnn_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = False
    opt.constrain_U = False
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 16:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = False
    opt.constrain_U = False
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 17:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 256
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 18:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
    opt.iterT = 5
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 19:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-2
    opt.iterT = 10
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 15:
    opt.optimizer = "irnn_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
    d = HAR_2(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = HAR_2(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = IndRNN(opt)
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

if opt.taskid == 920:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam"
    opt.clip_grad = True
    opt.gradclipvalue = 5
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.add_noise = True
    opt.endless_train = False
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
    opt.iterT = 5
    opt.endless_train = False
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
    opt.batch_size = 32
    opt.lr = 6e-4
    opt.iterT = 10
    opt.endless_train = False
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
    opt.iterT = 1
    opt.endless_train = False
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

if opt.taskid == 26:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 6e-4
    opt.iterT = 10
    opt.endless_train = False
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

if opt.taskid == 27:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
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

if opt.taskid == 28:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 1e-3
    opt.iterT = 5
    opt.endless_train = False
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

if opt.taskid == 29:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 1e-3
    opt.iterT = 10
    opt.endless_train = False
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

if opt.taskid == 30:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 6e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.add_noise = True
    opt.LSTM = True

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

if opt.taskid == 31:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 1e-3
    opt.iterT = 5
    opt.endless_train = False
    opt.add_noise = True
    opt.LSTM = True

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

if opt.taskid == 32:
    opt.optimizer = "FGSM_Adam"
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 2e-2
    opt.iterT = 10
    opt.endless_train = False
    opt.add_noise = True
    opt.LSTM = True

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
    
if opt.taskid == 34:
    opt.optimizer = "irnn_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = False
    opt.constrain_U = False
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 35:
    opt.optimizer = "irnn_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 36:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 32
    opt.lr = 2e-4
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = False
    opt.constrain_U = False
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 37:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 1e-3
    opt.iterT = 1
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 38:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 64
    opt.lr = 6e-4
    opt.iterT = 5
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



# -----------------------------------------------------------------------------------------------
    
if opt.taskid == 39:
    opt.optimizer = "FGSM_Adam"
    opt.num_layers = 6
    opt.hidden_size = 80
    opt.batch_size = 128
    opt.lr = 2e-2
    opt.iterT = 10
    opt.endless_train = False
    opt.constrain_grad = True
    opt.constrain_U = True
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
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


