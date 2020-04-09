import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F

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
        
        self.FC = nn.Linear(self.hidden_size, self.output_size)
        
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
        out, hidden = self.basic_rnn(X, hidden)    
        out = out[-1]
  
        out = self.FC(out)
        
        # out batch_size X n_output
        return out.view(-1, self.output_size), hidden 


# -------------------------------------------------------
# RNN structure forward step by step at each time sequence
# -------------------------------------------------------

class StepRNN(SimpleRNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(StepRNN, self).__init__(input_size, hidden_size, output_size)


    def forward(self, X, hidden, T):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)

        states = [(None, hidden)] # save previous and current hidden pairs

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
    def __init__(self, hidden_size, n_features, output_size, n_timesteps, sigma, v_dim, state_variant, alpha_val, K=1):
        super(ERNNCell, self).__init__(n_features, hidden_size, output_size)
        hiddenSize = hidden_size
        self.l1_in = nn.Linear(n_features, hidden_size)
        self.l1_hid = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hidden_size)
        # self.l_out = nn.Linear(hidden_size, inputSize)
        self.l_out = nn.Linear(hidden_size, output_size)
        self.eta1 = 1e-2
             
    def forward (self, x0, h0):
        Bsize,Fsize,Tsize = x0.shape
        Bsize, hiddensize = h0.shape
        out_tensor = []
        out_tensor.append(h0)

        # y = xp.zeros((Bsize,Fsize,0)).astype(np.float32)
        for ii in range(Tsize):
            last_h = out_tensor[-1]

            wx = self.l1_in(x0[:,:,ii]) # input->hidden_size
            
            #K=1
            uh0 = self.l1_hid(last_h+last_h) # hidden -> hidden
            F1h_hat = F.relu(uh0 + wx) 
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) 
            h1 = last_h + self.eta1 * (Fnh_hat - (last_h+last_h))
            h1 = F.relu(h1)
 
            out_tensor.append(h1)
        y1 = self.l_out(h1)
        return y1, h1
"""
class ERNNCell(SimpleRNN):
    def __init__(self, hidden_size, n_features, output_size, n_timesteps, sigma, v_dim, state_variant, alpha_val, K=1):
        print("ERNN -> ", state_variant)
        super(ERNNCell, self).__init__(n_features, hidden_size, output_size)
        self._hidden_size = hidden_size
        self.sigma = sigma
        self.v_dim  = v_dim
        self.t = 0
        self.state_variant = state_variant
        self.n_timesteps = n_timesteps
        self.eye_h = torch.eye(self._hidden_size)
        self.K = K#1
        self.gamma = 1

        # model_size = self._hidden_size * (self.v_dim + 1 + self._hidden_size + n_features) 
        # model_size *= 4.0
        # model_size /= 1024.0
        # print("model size = ", model_size, "KB")
        self.W = nn.Parameter(torch.Tensor(n_features, self._hidden_size))
        self.H = nn.Parameter(torch.Tensor(self._hidden_size, self._hidden_size))
        self.b = nn.Parameter(torch.Tensor(1, self._hidden_size))
        
        alpha_init = 0.01
        
        if alpha_val is not None: alpha_init = alpha_val

        self.alpha_init = alpha_init
        print('alpha_init = ', alpha_init)
        print('beta_init = ', beta_init)

        self.alpha = nn.Parameter(alpha_init * torch.ones((self.K, self.n_timesteps)))
        self.beta  = nn.Parameter(beta_init  * torch.ones((self.K, self.n_timesteps)))

    

    def NL(self, x):
        if self.sigma == 'relu': return F.relu(x)
        elif self.sigma == 'tanh': return F.tanh(x)
        elif self.sigma == 'sigmoid': return F.sigmoid(x)
        raise Exception('Non-linearity not found..')

    def forward(self, x, h, T):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        # x = x.permute(1, 0, 2)
        # h = h.permute(1, 0, 2)
        states = [(None, h)] # save previous and current hidden pairs

        # out_tensor = []
        batch_size, seq_len, fea_len = x.shape
        # out_tensor.append(h0)
        t = self.t
        U = self.H
        for i in range(0, seq_len):
            # last_h = out_tensor[-1]
            state = states[-1][1].detach()
            state.requires_grad=True

            h = state
            new_wx = torch.matmul(x[i], self.W)
            new_uh = torch.matmul(state, U)

            alpha = new_wx + self.b + new_uh
            # new_uh = tf.matmul(last_h, U)
            oldh = h
            for k in range(self.K):
                at = self.alpha[k][self.t] # #self.alpha_init 
                bt = self.beta[k][t] #(1-at)
                h = at*(self.NL(torch.matmul(h, U)+alpha) - oldh) + bt * h 
            # new_h = h
            new_state = h
            # out_tensor.append(new_h)
            states.append((state, new_state))
            
            while len(states) > T:
                    # Delete stuff that is too old
                    del states[0]
        output = h
        output = self.FC(output) # out batch_size X n_output
        return output.view(-1, self.output_size), states 
        # return torch.stack(out_tensor[1:]),torch.stack(out_tensor[1:])
        # output = output[-1]
        
 """   