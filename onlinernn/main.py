from exp.expConfig import ExpConfig
from onlinernn.options.train_options import TrainOptions
from onlinernn.datasets.mnist import MNIST, MNISTShift
from onlinernn.models.setting import Setting
from onlinernn.models.rnn_stopbp import StopBPRNN
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
        m = StopBPRNN(opt)
        p = ExpConfig(dataset=d, setting=s, model=m)
        p.run()



