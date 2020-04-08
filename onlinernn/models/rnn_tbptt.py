import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import StepRNN, TBPTTRNN
from onlinernn.models.rnn_stopbp import StopBPRNN

# -------------------------------------------------------
# Truncated BPTT 
# -------------------------------------------------------
class TBPTT(StopBPRNN):
    def __init__(self, opt):
        super(TBPTT, self).__init__(opt)

    def init_net(self):
        """
        Initialize model        
        """
        self.rnn_model = TBPTTRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()
        
        if (self.device.type == "cuda") and (self.opt.ngpu > 1):
           
            self.rnn_model = nn.DataParallel(self.rnn_model, list(range(self.opt.ngpu)))
        # get the initial W and U
        self.old_weights_ih = copy.deepcopy(self.rnn_model.basic_rnn.weight_ih_l0.data) 
        self.old_weights_hh = copy.deepcopy(self.rnn_model.basic_rnn.weight_hh_l0.data)

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # self.optimizer.zero_grad()

        self.init_states()
        outputs, loss = self.rnn_model(self.inputs, self.states, self.T, self.optimizer, self.criterion, self.labels)
        self.losses += loss
        self.train_acc += self.get_accuracy(outputs, self.labels, self.batch_size)
        # self.new_weights_ih = copy.deepcopy(self.rnn_model.basic_rnn.weight_ih_l0.data)
        # self.new_weights_hh = copy.deepcopy(self.rnn_model.basic_rnn.weight_hh_l0.data)
   
   # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states, self.T, None, None, None)
            outputs = outputs.to(self.device).detach()
            self.test_acc += self.get_accuracy(outputs, self.labels, self.batch_size)
            # self.save_result()
