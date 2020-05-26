import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import SimpleRNN, TBPTTRNN
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
        if self.opt.subsequene:
            self.rnn_model = SimpleRNN(self.opt).to(self.device)
        else:
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

    def set_input(self):
        self.inputs, self.labels = self.data
       
        self.inputs = self.inputs.view(-1, self.seq_len, self.input_size).to(self.device)
        self.batch_size = self.labels.shape[0]  # update batch 
        if self.opt.single_output:
            self.labels = self.labels.view(-1).to(self.device)
        else:
            self.labels = self.labels.to(self.device)
        if self.opt.predic_task in ['Binary', 'Logits']:
            self.labels = self.labels.float()

    def train_subsequence(self):
        """
        Reference: https://patrykchrabaszcz.github.io/when-truncated-bptt-fails/
        """
        losses = []
        nchunks = self.seq_len // self.opt.subseq_size
     
        for i in range(nchunks):
            
            sub_inputs, sub_labels = self.generate_subbatches(i, size=self.opt.subseq_size)
            if not self.opt.single_output:
                sub_labels = sub_labels.reshape(-1)
            self.optimizer.zero_grad()
            self.states = self.states.detach()
            outputs, self.states = self.rnn_model(sub_inputs, self.states)
           
            self.states = self.states.view(-1, self.num_layers, self.hidden_size)
           
            loss = self.criterion(outputs, sub_labels)      
            loss.backward()
            losses.append(loss.detach().item())
            if 'FGSM' in self.opt.optimizer:
          
                tbptt_first_iter = self.first_iter and i==0 # first iterT and first chunk in tbptt
                tbptt_last_iter = self.last_iter and i==nchunks-1 # last iterT and last chunk in tbptt
                # tbptt_first_iter = True
                # tbptt_last_iter = True
                self.optimizer.step(self.total_batches, tbptt_first_iter, tbptt_last_iter)
            else:
                self.optimizer.step()
        self.loss = sum(losses)/len(losses)
        self.outputs = outputs
        
    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        self.last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop
        self.init_states() 
        if self.opt.subsequene:
            self.train_subsequence()  
        else:
            self.outputs, self.loss = self.rnn_model(self.inputs, self.states, self.optimizer, self.total_batches, self.criterion, self.labels)

        if self.last_iter:
            # After last iterT, track Delta w, loss and acc
            # self.track_grad_flow(self.rnn_model.named_parameters())
            self.losses.append(self.loss)
            self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))

    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states)
            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))