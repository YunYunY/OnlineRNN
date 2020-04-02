import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from onlinernn.models.networks import SimpleRNN
from onlinernn.models.base_model import BaseModel

class VanillaRNN(BaseModel):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(VanillaRNN, self).__init__(opt)
        self.loss_method = "vanilla"
        self.optimizers = ([])  # define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.
        self.model_names = ["rnn_model"]
        self.time_step = 28
        self.n_layers = 1
        # self.reg_lambda = opt.reg_lambda

        # self.num_update = 1  # number of blocks to sample for each time step
        # self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)
        # self.mse_loss = torch.nn.MSELoss()
        # self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        # self._state = None

  
    def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        # self.state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        self.state = torch.zeros(self.n_layers, self.hidden_size)

    def init_net(self):
        """
        initialize model structure
        """
        self.rnn_model = SimpleRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.init_hidden()

    def init_loss(self):
        """
        Define Loss functions
        """
        self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)

    def init_optimizer(self):
        """
        Setup optimizers
        """
        self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)

    # ----------------------------------------------

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def setup(self):
        """
            Setup the schedulers or load the model
        """
        # TODO LR SCHEDULER MAY NEED TO BE STORED FOR RESUME
#https://discuss.pytorch.org/t/why-doesnt-resuming-work-properly-in-pytorch/19430/4
        if self.istrain:
            self.schedulers = [
                get_scheduler(optimizer, self.opt) for optimizer in self.optimizers
            ]
        # if not self.istrain:
        if not self.istrain or self.opt.continue_train:
            load_prefix = self.opt.load_iter if self.opt.load_iter > 0 else self.opt.epoch
            # load_prefix = "latest"
            # load_prefix = "30"
            print(f'Load the {load_prefix} epoch network')
            self.load_networks(load_prefix)

        self.print_networks(self.opt.verbose)
    
    # ----------------------------------------------
    def set_test_input(self):
        self.set_input()

    def set_input(self):
        # treat each row as a time_step, feature size is input_size
        self.inputs = self.data[0].view(self.batch_size, self.time_step, self.input_size).to(self.device)
        # permute inputs
        # self.inputs = self.inputs.permute(1, 0, 2)
        self.labels = self.data[1].to(self.device)

    def set_output(self):
        """
        Setup Lists to keep track of progress
        """
        self.losses = []

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        output, state = self.rnn_model.forward(self.inputs, self.state)
        self.loss = self.criterion(output, self.labels)
        self.loss.backward()
        self.optimizer.step()
        self.losses.append(self.loss.item())

   # ----------------------------------------------
    def save_losses(self, epoch):
        if self.opt.verbose:
            print( "Loss of epoch %d / %d "
                % (epoch, self.loss))
        np.savez(
            self.loss_dir + "/" + str(epoch) + "_losses.npz",
            losses=self.losses        )
