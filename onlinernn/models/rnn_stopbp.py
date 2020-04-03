import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from onlinernn.models.networks import SimpleRNN
from onlinernn.models.rnn_vanilla import VanillaRNN

# -------------------------------------------------------
# FPP reference https://openreview.net/pdf?id=SJgmR0NKPr
# -------------------------------------------------------
class SoptBPRNN(VanillaRNN):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(SoptBPRNN, self).__init__(opt)
        # self.mse_loss = torch.nn.MSELoss()
        # self._state = None

    # def initial_states(self):
    #     self.states = torch.zeros(size = [self.num_layers, self.batch_size, self.hidden_size], requires_grad=True).to(self.device)
 

    def set_input(self):
        # treat each row as a seq_len, feature size is input_size
        self.inputs, self.labels = self.data
        self.inputs = self.inputs.view(28, -1, 28).to(self.device)
        self.labels = self.labels.to(self.device)
        # update batch 
        self.batch_size = self.labels.shape[0]

    def initial_states(self):
        self.states = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer.zero_grad()
        self.initial_states()
        states_queue = [(None, self.states)]
        # loop timestep
        for j, inp in enumerate(self.inputs):
            # get the latest state
            state = states_queue[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.rnn_model(inp.view(self.batch_size, 1, 28), state)
            output, new_state = output.to(self.device), new_state.to(self.device)
            states_queue.append((state, new_state))
            while len(states_queue) > self.T:
                # Delete stuff that is too old
                del states_queue[0]
        # calculate loss in the last timestep 
        self.loss = self.criterion(output, self.labels)
        self.loss.backward(retain_graph=True)
        for k in range(self.T-1):
            # if we get all the way back to the "init_state", stop
            if states_queue[-k-2][0] is None:
                break
            curr_grad = states_queue[-k-1][0].grad
            states_queue[-k-2][1].backward(curr_grad, retain_graph=True)
        self.optimizer.step()
        self.losses += self.loss.detach().item()
        self.train_acc += self.get_accuracy(output, self.labels, self.batch_size)
