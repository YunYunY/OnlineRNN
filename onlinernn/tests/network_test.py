import torch
from torchsummary import summary
from onlinernn.models.networks import SimpleRNN, ERNNCell
from onlinernn.datasets.mnist import MNIST, MNISTShift
from onlinernn.options.train_options import TrainOptions



# ------------------------------------------------------------

opt = TrainOptions().parse()
opt.istrain = True
opt.device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
)
opt.download_data = False

# -------------------------------------------------------

def test_SimpleRNN():
    input_size, hidden_size, output_size = 28, 512, 10
    rnn = SimpleRNN(input_size, hidden_size, output_size)   
    # visually check the model structure
    print(rnn)


# -------------------------------------------------------
def test_ERNNCell():
    rnn = ERNNCell(256, 28,  10, 28, 'cpu', 'relu',32, 46, alpha_val=0.001, K=1)
    print(rnn)

