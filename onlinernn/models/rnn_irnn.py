import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import StepRNN, ERNNCell
from onlinernn.models.rnn_vanilla import VanillaRNN

# -------------------------------------------------------
# iRNN Reference https://openreview.net/pdf?id=HylpqA4FwS 
# -------------------------------------------------------
class IRNN(VanillaRNN):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(IRNN, self).__init__(opt)

        self.mse_loss = torch.nn.MSELoss()

    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        self.rnn_model = ERNNCell(self.hidden_size, self.input_size, self.output_size, 
                                self.seq_len, self.device, 'relu',32, 46, alpha_val=0.001, K=1).to(self.device)

        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()
        if (self.device.type == "cuda") and (self.opt.ngpu > 1):
            print('Run parallel')
            self.rnn_model = nn.DataParallel(self.rnn_model, list(range(self.opt.ngpu)))

    def init_states(self):
        # self.states = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        self.states = torch.zeros(self.batch_size, self.hidden_size).to(self.device)

    def set_input(self):
        self.inputs, self.labels = self.data
        # sequence MNIST
        # self.seq_len = self.seq_len * self.input_size # 784
        # self.input_size = 1 # input 1 pixel each time
        self.inputs = self.inputs.view(-1, self.input_size, self.seq_len).to(self.device)
        self.labels = self.labels.to(self.device)
        # if self.permute_row:
            # self.inputs = self.inputs[:, :, self.permute_idx].to(self.device)
        # update batch 
        self.batch_size = self.labels.shape[0]