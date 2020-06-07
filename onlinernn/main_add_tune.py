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

# torch.manual_seed(42)
# np.random.seed(42)

torch.manual_seed(1024)
np.random.seed(1024)
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

opt.N_TRAIN  = 50000
opt.N_TEST = 2000
opt.seq_len = 100
opt.iterT = 1
opt.predic_task = 'Logits'
opt.test_batch = True

if opt.taskid == 100:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")

    opt.U_bound = 1.00695
    opt.optimizer = "Adam" #"FGSM_Adam"
    opt.num_layers = 2
    opt.hidden_size = 128
    opt.batch_size = 50
    input_size = 1
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 26
    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = IndRNN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 200:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.LSTM = True
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 2
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.weight_decay = 1e-3

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
    # m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()





if opt.taskid == 201:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam" 
    opt.LSTM = True
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 2
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.weight_decay = 1e-3

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
    # m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 500:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-3

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 501:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-2

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 502:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-1

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 503:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 504:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-4

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 704:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 299
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-4
    opt.iterT = 10

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()





if opt.taskid == 706:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 299
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0
    opt.iterT = 10

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 705:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 299
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 2e-5
    opt.iterT = 10

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 804:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-4
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 809:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-3
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 810:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-2
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 811:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 2e-3
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 812:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 4e-3
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 813:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 6e-3
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 808:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 806:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 2
    opt.lrgamma = 0.8
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-4
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 807:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4
    # opt.niter_decay = 2
    # opt.lrgamma = 0.8
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 2e-4
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 805:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 2e-5
    opt.iterT = 5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 904:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-4

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 906:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 907:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 908:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 909:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 910:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 0

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 905:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 38
    # opt.subsequene = True
    # opt.subseq_size = 100
    opt.weight_decay = 2e-5

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
    # m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 604:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-2
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-3

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 605:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-1
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-2

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()


if opt.taskid == 606:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "FGSM_Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 2e-4
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 2e-5

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()

if opt.taskid == 401:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 18
    opt.subsequene = True
    opt.subseq_size = 10
    opt.weight_decay = 1e-3

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()



if opt.taskid == 402:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-3

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()




if opt.taskid == 403:
    print(f"----------------- Inside iteration T is {opt.iterT} -----------------")
    opt.optimizer = "Adam" 
    opt.hidden_size = 128
    opt.batch_size = 50 # 50
    opt.lr = 1e-3
    opt.niter_decay = 0
    opt.lrgamma = 0.1
    opt.endless_train = False
    opt.niter = 99
    opt.subsequene = True
    opt.subseq_size = 100
    opt.weight_decay = 1e-4

    d = ADDING(opt)

    # train and eval in every epoch 
    if opt.eval_freq > 0 and opt.istrain:
        opt.istrain = False
        d_test = ADDING(opt)
        opt.istrain = True
    else:
        d_test = None 
    s = RNN(opt)
    # m = VanillaRNN(opt)
    m = TBPTT(opt)
    p = ExpConfig(dataset=d, setting=s, model=m, dataset_test=d_test)
    p.run()







