from exp.expConfig import ExpConfig
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTShift
from onlinernn.datasets.har import HAR_2
from onlinernn.models.setting import Setting
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.rnn_stopbp import StopBPRNN
from onlinernn.models.rnn_tbptt import TBPTT
from onlinernn.models.rnn_irnn import IRNN
from onlinernn.models.setting import RNN

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

if opt.taskid == 0:
    for T in opt.T:
        print(f"----------------- Truncation T is {T} -----------------")
        opt.T_ = T

        d = MNIST(opt)
        s = RNN(opt)
        m = IRNN(opt)
        p = ExpConfig(dataset=d, setting=s, model=m)
        p.run()
# -----------------------------------------------------------------------------------------------
# Vanishing Gradient Solver FGSM
# -----------------------------------------------------------------------------------------------

if opt.taskid == 1:
    for T in opt.T:
        print(f"----------------- Inside iteration T is {T} -----------------")
        opt.T_ = T
        d = HAR_2(opt)
        # d = MNIST(opt)
        s = RNN(opt)
        m = VanillaRNN(opt)
        p = ExpConfig(dataset=d, setting=s, model=m)
        p.run()



