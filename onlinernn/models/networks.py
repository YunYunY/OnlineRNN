import math
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler


# -------------------------------------------------------
# Basic RNN structure
# -------------------------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self, opt):
        super(SimpleRNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.input_size = opt.feature_shape
        self.output_size = opt.n_class
        self.seq_len = opt.seq_len
        self.basic_rnn = nn.RNN(self.input_size, self.hidden_size)
        
        # self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        

        if opt.predic_task in ['Binary', 'Logits']:
            self.FC = nn.Linear(self.hidden_size, 1)
        else:
            self.FC = nn.Linear(self.hidden_size, self.output_size)
            

    def last_layer(self):
        if self.opt.predic_task == 'Binary':
            # To make model structure identity with paper iRNN
            out = torch.sigmoid(self.FC(self.hidden_final))
            return out.view(-1), self.hidden_final
        elif self.opt.predic_task == 'Logits':
            out = self.FC(self.out[-1])
            # out = torch.tanh(self.FC(self.out[-1]))
            return out.view(-1), self.hidden_final
        else:
            if self.opt.single_output:
                out = self.out[-1]
                out = self.FC(out)
                # out batch_size X n_output
                return out.view(-1, self.output_size), self.hidden_final
            else:
                out = self.out.permute(1, 0, 2)
                out = self.FC(out)
               
                return out.reshape(-1, self.output_size), self.hidden_final


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
        # self.out, self.hidden_final = self.lstm(X)  
        self.out, self.hidden_final = self.basic_rnn(X, hidden)  
       
        return self.last_layer()
 

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
        # transforms X to dimensions: x: (seq_len, batch, input_size) 
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
    def __init__(self, opt):
        super(TBPTTRNN, self).__init__(opt)
        self.T1 = opt.for_trunc
        self.T2 = opt.back_trunc
        self.retain_graph = self.T1 < self.T2
        
    def forward(self, X, hidden, optimizer=None, total_batches=None, criterion=None, labels=None):
        """
        Args:
            X: Input Sequential data
            hidden: Initialized hidden layer
        """
        losses = 0
        nloss = 0
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        # transforms X to dimensions: x: (seq_len, batch, input_size) 
        X = X.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        states = [(None, hidden)] # save previous and current hidden pairs

        if labels == None: # model test
            for i in range(seq_len):
                inp = X[i].clone().view(1, batch_size, self.input_size)
                state = states[-1][1].detach()
                state.requires_grad=True
                output, new_state = self.basic_rnn(inp, state)    
                states.append((state, new_state))
                while len(states) > 2:
                        # Delete stuff that is too old
                        del states[0]
            self.out, self.hidden_final = output, new_state
            # output = output[-1]
            # output = self.FC(output)
            return self.last_layer()
            # return output.view(-1, self.output_size), states 

        last_iter = (total_batches-1)%self.opt.iterT == (self.opt.iterT-1) # if this is the last step of inner loop
        first_iter = (total_batches-1)%self.opt.iterT == 0 # if this is the first iter of inner loop

        # every T1 steps backpropagate taking T2 back steps
        for j in range(seq_len):
            inp = X[j].clone().view(1, batch_size, self.input_size)
            state = states[-1][1].detach()
            state.requires_grad=True
      
            output, new_state = self.basic_rnn(inp, state)    
            states.append((state, new_state))
            # output = output[-1]
            # output = self.FC(output) 
            self.out, self.hidden_final = output, new_state
      
            output, _ =  self.last_layer()
            
            '''
            # check the gradient calculation step
            def get_pr(idx_val):
                    def pr(*args):
                        print("doing backward {}".format(idx_val))
                    return pr
            new_state.register_hook(get_pr(j))
            output.register_hook(get_pr(j))
            print("doing fw {}".format(j))
            '''

            while len(states) > self.T2:
                    # Delete stuff that is too old
                    del states[0] 
            if (j+1)% self.T1 == 0: # start backward after T1 forward steps 
                # FGSM flag, first batch in inner loop and first 
                initial = first_iter and j+1 == self.T1
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
               
                if 'FGSM' in self.opt.optimizer:
                    if self.opt.iterB > 0:
                        optimizer.step(total_batches, sign_option=True)
                    else:
                        optimizer.step(total_batches)
                else:
                    optimizer.step()

        # out batch_size X n_output
        return output, losses/nloss
       

# -------------------------------------------------------
# ERNN reference: https://openreview.net/pdf?id=HylpqA4FwS
# -------------------------------------------------------


class ERNNCell(nn.Module):

    def __init__(self, device, hidden_size, n_features, n_timesteps, sigma, v_dim, alpha_val, K=1):
        super(ERNNCell, self).__init__()
        self._hidden_size = hidden_size
        self.sigma = sigma
        self.v_dim  = v_dim
        self.t = 0
        self.n_timesteps = n_timesteps
        self.eye_h = torch.eye(self._hidden_size).to(device)
        self.K = K#1
        self.gamma = 1

        model_size = self._hidden_size * (self.v_dim + 1 + self._hidden_size + n_features) 
        model_size *= 4.0
        model_size /= 1024.0
        print("model size = ", model_size, "KB")
        self.W = nn.Parameter(torch.Tensor(n_features, self._hidden_size))
        self.H = nn.Parameter(torch.Tensor(self._hidden_size, self._hidden_size))
        self.b = nn.Parameter(torch.Tensor(self._hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.v_dim, self._hidden_size))
        self.V2 = nn.Parameter(torch.Tensor(self._hidden_size, self.v_dim))
        
        
        alpha_init = 0.001
        
        if alpha_val is not None: alpha_init = alpha_val

        self.alpha_init = alpha_init
        beta_init = 1.0 - alpha_init
        print('alpha_init = ', alpha_init)
        print('beta_init = ', beta_init)

        self.alpha = nn.Parameter(alpha_init * torch.ones((self.K, self.n_timesteps)))
        self.beta  = nn.Parameter(beta_init  * torch.ones((self.K, self.n_timesteps)))
        
        stdv = 1.0 / math.sqrt(self._hidden_size)
        for name,weight in self.named_parameters():
            if 'alpha' not in name and 'beta' not in name:
                nn.init.uniform_(weight, -stdv, stdv)
        
    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size        

    def NL(self, x):
        if self.sigma == 'relu': return torch.relu(x)
        elif self.sigma == 'tanh': return torch.tanh(x)
        elif self.sigma == 'sigmoid': return torch.sigmoid(x)
        raise Exception('Non-linearity not found..')

    def forward(self, x, h):
        
        VV2 = torch.matmul(self.V2, self.V)
        infite2 = VV2 + self.eye_h
        
        out_tensor = []
        seq_len,batch_size, fea_len = x.shape
        out_tensor.append(h)

        for i in range(0,seq_len):
            last_h = out_tensor[-1]
            h = last_h
            new_wx = torch.matmul(x[i], self.W)
            P, U = infite2, infite2

            new_uh = torch.matmul(h, U)
            alpha = new_wx + self.b + new_uh
            
            oldh = h
            for k in range(self.K):
                at = self.alpha[k][self.t]
                bt = (1-at)
                h = at*(self.NL(torch.matmul(torch.matmul(h, U)+alpha,P)) - oldh) + bt * h 
            new_h = h
            out_tensor.append(new_h)
        return new_h


class IRNN_model(SimpleRNN):
    def __init__(self, opt):
        super(IRNN_model, self).__init__(opt)
        
        self.layers = nn.ModuleList([])

        self.layers.append(ERNNCell(opt.device, self.hidden_size, self.input_size, opt.seq_len, sigma='relu', v_dim=20, alpha_val=0.001, K=3)) 
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        inp = x
        cnt = 0
        for layer in self.layers:
            h0 = torch.zeros(x.shape[1], self.hidden_size).to(self.opt.device)
            y = layer(inp,h0)

        y_final = self.linear(y)
        #final = F.softmax(y_final,dim=1)
        return y_final


'''
class ERNNCell(SimpleRNN):
     
    # ERNN with K=1
  
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

'''

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
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.niter_decay, gamma=opt.lrgamma)
    return scheduler
