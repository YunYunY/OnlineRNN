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


    # def forward(self, X, hidden, T):
    #     batch_size = X.shape[0]
    #     seq_len = X.shape[1]
    #     # transforms X to dimensions: n_steps X batch_size X n_inputs
    #     X = X.permute(1, 0, 2)
    #     hidden_ = [hidden] 
        # for i in range(X.shape[0]):
        #     old_hidden = hidden_[-1].clone()
        #     inp = X[i].clone().view(1, batch_size, self.input_size)    
        #     if seq_len - i == T:
        #         old_hidden.detach()
        # # self.basic input:
        # #     x: (seq_len, batch, input_size)
        # #     state: (num_layers, batch, hidden_size)
        # # self.basic output 
        # #     output: (seq_len, batch, hidden_size)
        # #     state: (num_layers, batch, hidden_size)
        #     out, new_hidden = self.basic_rnn(inp, old_hidden)    
        #     hidden_.append(new_hidden)
        # out = out[-1]
        # out = self.FC(out)
        
        # # out batch_size X n_output
        # return out.view(-1, self.output_size), hidden 
    