import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# -------------------------------------------------------
# RNN
# -------------------------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.n_neurons = hidden_size
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)
        
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        
    def forward(self, X, hidden):

        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2) 
         
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
        return out.view(-1, self.n_outputs), hidden 

