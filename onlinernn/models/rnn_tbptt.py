import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import StepRNN, TBPTTRNN
# from onlinernn.models.rnn_stopbp import StopBPRNN
from onlinernn.models.rnn_vanilla import VanillaRNN
# -------------------------------------------------------
# Truncated BPTT 
# -------------------------------------------------------
class TBPTT(VanillaRNN):
    def __init__(self, opt):
        super(TBPTT, self).__init__(opt)

    def init_net(self):
        """
        Initialize model        
        """
        self.rnn_model = TBPTTRNN(self.opt).to(self.device)
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
        first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop
        self.init_states()  
        self.outputs, self.loss = self.rnn_model(self.inputs, self.states, self.optimizer, self.total_batches, self.criterion, self.labels)

        if last_iter:
            # After last iterT, track Delta w, loss and acc
            self.track_grad_flow(self.rnn_model.named_parameters())
            self.losses.append(self.loss)
            self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))
           

    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states)
            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))