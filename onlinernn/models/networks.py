import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

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
       
    