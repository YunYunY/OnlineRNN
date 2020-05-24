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

torch.manual_seed(42)
np.random.seed(42)
torch.set_printoptions(precision=8)

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

opt.adding_train = 30000
opt.adding_test = 100
opt.seq_len = 100
opt.iterT = 30
opt.predic_task = 'Logits'

opt.verbose_batch = True

if opt.taskid == 100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" # "RMSprop"
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4 #1e-3
    opt.niter_decay = 1
    opt.lrgamma = 0.9
    opt.endless_train = False
    opt.niter = 9
    opt.endless_train = True
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