from options.train_options import TrainOptions
from datasets.mnist import MNISTShift
from models.setting import Setting
from models.gan_model import StopBPTT
from models.setting import RNN
from exp.expConfig import ExpConfig

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
    opt.num_test = 1  # how many test batches to run
# -----------------------------------------------------------------------------------------------
if opt.taskid == 0:
    d = MNIST(opt)
    s = GAN(opt)
    m = VanillaGAN(opt)
    p = ExpConfig(dataset=d, setting=s, model=m)
    p.run()





