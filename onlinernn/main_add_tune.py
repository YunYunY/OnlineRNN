import torch
import numpy as np
from exp.expConfig import ExpConfig
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTPixel, MNISTPixelPermute, MNISTShift
from onlinernn.datasets.mnist_byte import MNIST_byte
from onlinernn.datasets.har import HAR_2
from onlinernn.datasets.dsa import DSA_19
from onlinernn.datasets.adding import ADDING
from onlinernn.models.setting import Setting
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.rnn_stopbp import StopBPRNN
from onlinernn.models.rnn_tbptt import TBPTT
from onlinernn.models.rnn_irnn import IRNN
from onlinernn.models.rnn_ind import IndRNN
from onlinernn.models.setting import RNN

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)

# torch.manual_seed(1024)
# np.random.seed(1024)
# torch.set_printoptions(precision=8)

# -----------------------------------------------------------------------------------------------
# Get training options
# -----------------------------------------------------------------------------------------------
opt = TrainOptions().parse()
# Hardcode some parameters for test
if not opt.istrain:
    opt.num_threads = 0  # test code only supports num_threads = 1
    # opt.num_test = 1  # how many test batches to run
# -----------------------------------------------------------------------------------------------
# self.ap_timesteps=[100, 200, 400, 750]
# self.ap_samples=[30000, 50000, 40000, 100000]

opt.N_TRAIN  = 50000
opt.N_TEST = 2000
opt.seq_len = 200
opt.iterT = 1
opt.predic_task = 'Logits'
opt.test_batch = True


if opt.taskid == 0:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 1:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 2:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 101:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 101:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 111:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
    


if opt.taskid == 116:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
    

if opt.taskid == 200:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 201:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 202:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 203:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 2151:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 2161:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 300:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 301:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 302:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 303:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 3151:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 3161:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 400:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 500:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 503:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.meta = 1
    opt.task = 'GD'
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 600:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 603:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.meta = 1
    opt.task = 'HB'
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 700:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 703:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.meta = 1
    opt.task = 'NAG'
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 800:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 803:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.meta = 2
    opt.task = 'GD'
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 903:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.meta = 2
    opt.task = 'HB'
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 1000:
    opt.iterT = 1
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 1001:
    opt.iterT = 1
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 1002:
    opt.iterT = 1
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 1103:
    opt.meta = 2
    opt.task = 'NAG'
    opt.iterT = 1
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.seq_len = 200
    opt.num_layers = 1
    opt.hidden_size = 128
    opt.batch_size = 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 100
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    m = VanillaRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()
