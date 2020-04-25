import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import StepRNN, ERNNCell
from onlinernn.models.rnn_vanilla import VanillaRNN

# -------------------------------------------------------
# Stop BPTT 
# -------------------------------------------------------
class StopBPRNN(VanillaRNN):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(StopBPRNN, self).__init__(opt)
        self.mse_loss = torch.nn.MSELoss()
        self.reg_lambda = opt.reg_lambda


    # def initial_states(self):
    #     self.states = torch.zeros(size = [self.num_layers, self.batch_size, self.hidden_size], requires_grad=True).to(self.device)
 
    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        # self.rnn_model = StepRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.rnn_model = ERNNCell(self.hidden_size, self.input_size, self.output_size, 
                                100, 'relu',32, 49, alpha_val=0.001, K=1 ).to(self.device)

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
        self.optimizer.zero_grad()
        self.init_states()
    
        outputs, states = self.rnn_model(self.inputs, self.states, self.T)
        outputs = outputs.to(self.device)
        # outputs, state = outputs.to(self.device), state.to(self.device)
       
        self.loss = self.criterion(outputs, self.labels)

        # add regularization
        if self.iload>0: 
            reg1 = torch.sqrt(self.mse_loss(self.old_weights_ih, self.new_weights_ih))
            reg2 = torch.sqrt(self.mse_loss(self.old_weights_hh, self.new_weights_hh))
            # print(self.loss)
            # print(reg1)
            # print(reg2)
            self.losses += self.loss.item()
            self.reg1 += reg1.item()
            self.reg2 += reg2.item()

            self.loss += self.reg_lambda * (reg1 + reg2)
            self.old_weights_ih = self.new_weights_ih    
            self.old_weights_hh = self.new_weights_hh

        self.loss.backward(retain_graph=False)

        for i in range(self.T-1):
                # if we get all the way back to the "init_state", stop
                if states[-i-2][0] is None:
                    break
                curr_grad = states[-i-1][0].grad
                states[-i-2][1].backward(curr_grad, retain_graph=False)
        self.optimizer.step()
        # self.losses += self.loss.detach().item()
        self.train_acc += self.get_accuracy(outputs, self.labels, self.batch_size)
        self.new_weights_ih = copy.deepcopy(self.rnn_model.basic_rnn.weight_ih_l0.data)
        self.new_weights_hh = copy.deepcopy(self.rnn_model.basic_rnn.weight_hh_l0.data)

    def save_losses(self, epoch):
        # if self.opt.verbose:
        #     print( "Loss of epoch %d / %d "
        #         % (epoch, self.loss))
        np.savez(
            self.loss_dir + "/" + str(epoch) + "_losses.npz",
            losses=self.losses, reg1=self.reg1, reg2=self.reg2     )

    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states, self.T)
            outputs = outputs.to(self.device).detach()
            self.test_acc += self.get_accuracy(outputs, self.labels, self.batch_size)
            # self.save_result()
