import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler

# -------------------------------------------------------
# Basic RNN structure
# -------------------------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.basic_rnn = nn.RNN(self.input_size, self.hidden_size)
        # self.FC = nn.Linear(self.hidden_size, self.output_size)
        self.FC = nn.Linear(self.hidden_size, 1)

    def forward(self, X, hidden):

        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2) 
        hidden = hidden.permute(1, 0, 2)
        # self.basic input:
        #     x: (seq_len, batch, input_size)
        #     state: (num_layers, batch, hidden_size)
        # self.basic output 
        #     output: (seq_len, batch, hidden_size)
        #     state: (num_layers, batch, hidden_size)
        out, hidden_final = self.basic_rnn(X, hidden)  
        
        # out = out[-1]
        # out = self.FC(out)
        # To make model structure identity with paper iRNN
        out = torch.sigmoid(self.FC(hidden_final))


        # out batch_size X n_output
        # return out.view(-1, self.output_size), hidden_final
        return out.view(-1), hidden_final

# -------------------------------------------------------
# RNN structure forward step by step at each time sequence
# -------------------------------------------------------

class StepRNN(SimpleRNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(StepRNN, self).__init__(input_size, hidden_size, output_size)


    # def forward(self, X, hidden, T):
    def forward(self, X, hidden):

        batch_size = X.shape[0]
        seq_len = X.shape[1]
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)

        states = [(None, hidden)] # save previous and current hidden pairs

        for i in range(X.shape[0]):
            inp = X[i].clone().view(1, batch_size, self.input_size)
            state = states[-1][1]
            # state = states[-1][1].detach()
            # state.requires_grad=True
            output, new_state = self.basic_rnn(inp, state)    
            if i == 0:
                state_start = new_state
            if i == (X.shape[0]-1):
                state_final = new_state
            states.append((state, new_state))
            # while len(states) > T:
                    # Delete stuff that is too old
                    # del states[0]
        output = output[-1]
        output = self.FC(output)
        
        # out batch_size X n_output
        return output.view(-1, self.output_size), states, state_start, state_final 



# -------------------------------------------------------
# TBPTTRNN 
# -------------------------------------------------------

class TBPTTRNN(SimpleRNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(TBPTTRNN, self).__init__(input_size, hidden_size, output_size)

    def forward(self, X, hidden, T, optimizer, criterion, labels):
        """
        Args:
            X: Input Sequential data
            hidden: Initialized hidden layer
            T1: forward step parameter in TBPTT
            T2: backward step parameter in TBPTT
        """
        losses = 0
        nloss = 0
        self.T1 = self.T2 = T
        self.retain_graph = self.T1 < self.T2
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        states = [(None, hidden)] # save previous and current hidden pairs

        if labels == None: # test
            for i in range(X.shape[0]):
                inp = X[i].clone().view(1, batch_size, self.input_size)
                state = states[-1][1].detach()
                state.requires_grad=True
                output, new_state = self.basic_rnn(inp, state)    
                states.append((state, new_state))
                while len(states) > T:
                        # Delete stuff that is too old
                        del states[0]
            output = output[-1]
            output = self.FC(output)
            
            # out batch_size X n_output
            return output.view(-1, self.output_size), states 


        # every T1 steps backpropagate taking T2 back steps

        for j in range(X.shape[0]):
            inp = X[j].clone().view(1, batch_size, self.input_size)
            state = states[-1][1].detach()
            state.requires_grad=True
      
            output, new_state = self.basic_rnn(inp, state)    
            states.append((state, new_state))
            output = output[-1]
            output = self.FC(output) 

            # check the gradient calculation step
            def get_pr(idx_val):
                    def pr(*args):
                        print("doing backward {}".format(idx_val))
                    return pr
            new_state.register_hook(get_pr(j))
            output.register_hook(get_pr(j))
            print("doing fw {}".format(j))


            while len(states) > self.T2:
                    # Delete stuff that is too old
                    del states[0] 
            if (j+1)% self.T1 == 0:
                optimizer.zero_grad()
                # output.view(-1, self.output_size)
                loss = criterion(output, labels)
                losses += loss.item()
                nloss += 1
                # backprop last module (keep graph only if they ever overlap)
                loss.backward(retain_graph=self.retain_graph)
                for i in range(self.T2-1):
                    # if we get all the way back to the "init_state", stop
                    if states[-i-2][0] is None:
                        break
                    curr_grad = states[-i-1][0].grad
                    states[-i-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                optimizer.step()
        

        # out batch_size X n_output
        return output, losses/nloss
       

# -------------------------------------------------------
# ERNN reference: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------
class ERNNCell(SimpleRNN):
    '''
    ERNN with K=1
    '''
    def __init__(self, hidden_size, n_features, output_size, n_timesteps, device, sigma, v_dim, state_variant, alpha_val, K=1):
        super(ERNNCell, self).__init__(n_features, hidden_size, output_size)
        hiddenSize = hidden_size # set it to equal for now
        self.device = device
        self.l1_in = nn.Linear(n_features, hidden_size)
        self.l1_hid = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hidden_size)
        self.l_out1 = nn.Linear(hidden_size, n_features)
        self.l_out2 = nn.Linear(n_features * n_timesteps, output_size)

        self.l_out = nn.Linear(hidden_size, output_size)
        self.eta1 = 1e-2
             
    def forward (self, x0, h0):

        Bsize,Fsize,Tsize = x0.shape
        Bsize, hiddensize = h0.shape
        out_tensor = []
        out_tensor.append(h0)
        # print(x0.shape)
        y = torch.zeros(Bsize, Fsize, 0).to(self.device)
        for ii in range(Tsize):
            last_h = out_tensor[-1]

            wx = self.l1_in(x0[:,:,ii]) # input->hidden_size
            
            #K=1
            uh0 = self.l1_hid(last_h+last_h) # hidden -> hidden
            F1h_hat = F.relu(uh0 + wx) 
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) 
            # Fnh_hat = F1h_hat
            h1 = last_h + self.eta1 * (Fnh_hat - (last_h+last_h))
            
            # h1 = F.relu(h1)
            # y1 = torch.sigmoid(self.l_out1(h1))
            # y1 = y1[:,:,None]
            # y = torch.cat((y,y1), axis=2)

            out_tensor.append(h1)
        
        # y1 = self.l_out2(y.view(Bsize, -1))
        y1 = self.l_out(h1)
        return y1, h1



# -----------------------------------------------------
# functions for learning rate schedule
# ------------------------------------------------------


def create_lambda_rule(opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
        return lr_l

    return lambda_rule


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags　
                                  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    if opt.lr_policy == "linear":
        lr_lambda = create_lambda_rule(opt)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.niter_decay, gamma=0.1)
    return scheduler
