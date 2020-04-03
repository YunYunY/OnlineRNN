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
        self.seq_len = 28
 
        # self.reg_lambda = opt.reg_lambda

        # self.num_update = 1  # number of blocks to sample for each time step
        # self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)
        # self.mse_loss = torch.nn.MSELoss()
        # self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        # self._state = None

  
    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        self.rnn_model = SimpleRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()



    def init_loss(self):
        """
        Define Loss functions
        """
        self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)

    def init_optimizer(self):
        """
        Setup optimizers
        """
        # self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        self.optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=self.lr)
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
        # treat each row as a seq_len, feature size is input_size
        self.inputs, self.labels = self.data
        self.inputs = self.inputs.view(-1, 28,28).to(self.device)
        self.labels = self.labels.to(self.device)
        # update batch 
        self.batch_size = self.labels.shape[0]

    def set_output(self):
        self.losses = 0
        self.train_acc = 0

    # -------------------------------------------------------
    # Initialize first hidden state      

    def initial_states(self):
        self.states = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer.zero_grad()
        self.initial_states()
        outputs, state = self.rnn_model(self.inputs, self.states)
        outputs, state = outputs.to(self.device), state.to(self.device)
        self.loss = self.criterion(outputs, self.labels)
        self.loss.backward()
        self.optimizer.step()
        self.losses += self.loss.detach().item()
        self.train_acc += self.get_accuracy(outputs, self.labels, self.batch_size)


    def training_log(self, i, epoch):
        """
        Create log
        """
        print(
            f"[{epoch}/{self.opt.n_epochs}], {i}, {self.datasize}, Loss: {self.loss.detach().item()}"
        )

   # ----------------------------------------------

    def get_accuracy(self, logit, target, batch_size):
        """ 
        Obtain accuracy for training round 
        """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()

    def save_losses(self, epoch):
        if self.opt.verbose:
            print( "Loss of epoch %d / %d "
                % (epoch, self.loss))
        np.savez(
            self.loss_dir + "/" + str(epoch) + "_losses.npz",
            losses=self.losses        )
