import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import IRNN_model
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
        self.rnn_model = IRNN_model(self.opt).to(self.device)

        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()
        if (self.device.type == "cuda") and (self.opt.ngpu > 1):
            print('Run parallel')
            self.rnn_model = nn.DataParallel(self.rnn_model, list(range(self.opt.ngpu)))

    def set_input(self):
        self.inputs, self.labels = self.data
        self.inputs = self.inputs.permute(1,0,2).to(self.device)

        # self.inputs = self.inputs.view(-1, self.seq_len, self.input_size).to(self.device)
      
        self.labels = self.labels.view(-1).to(self.device)
        if self.opt.predic_task == 'Binary':
            self.labels = self.labels.float()
        # update batch 
        self.batch_size = self.labels.shape[0]

    def forward(self):
        """
        Forward path 
        """
        self.optimizer.zero_grad()
        outputs = self.rnn_model(self.inputs)
        self.outputs = outputs.to(self.device)
 
    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.backward()
        first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop
        if 'FGSM' in self.opt.optimizer:
            # if self.opt.iterB > 0:
            #     self.optimizer.step(self.total_batches, sign_option=True)
            # else:
            self.optimizer.step(self.total_batches)
        else:
            self.optimizer.step()

        if last_iter:
            # After last iterT, track Delta w, loss and acc
            # self.track_grad_flow(self.rnn_model.named_parameters())
            self.losses.append(self.loss.detach().item())
            self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))


    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            outputs = self.rnn_model(self.inputs)
            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))