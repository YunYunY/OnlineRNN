import torch
import torch.nn as nn

# -------------------------------------------------------
# RNN
# -------------------------------------------------------

class SimpleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
            super(SimpleRNN, self).__init__()
            self.n_layers = 1
            self.hidden_size = hidden_size
            self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        """
        Args:
            x: (T, batch_size, output_size)
            state: (layer = 1, batch_size, hidden_size)

        Outputs:
            output: (T, batch_size, output_size)
            state: (1, batch_size, hidden_size)
        """
        output, state = self.rnn_layer(x, state)  # output = (T, batch_size, hidden_size)

        output = self.output_layer(output)
    
        return output, state


  def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        # self.state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        self.state = torch.zeros(self.n_layers, self.hidden_size)

        
   ModeZeros = 'Zeros'
    ModeRandom = 'Random'
def initial_state(mode, num_layers, hidden_size):
    if mode == "Zeros":
        return np.zeros((num_layers, self.hidden_size), dtype=np.float32)
    elif mode == self.ModeRandom:
        random_state = np.random.normal(0, 1.0, (self.num_layers, self.hidden_size))
        return np.clip(random_state, -1, 1).astype(dtype=np.float32)
    else:
        raise RuntimeError('No mode %s' % mode)

def initial_states(self, mode, samples=64):
    states = [self.initial_state(mode) for _ in range(samples)]
    states = np.stack(states)
    states = np.swapaxes(states, 1, 0)
    states = Variable(torch.from_numpy(states), requires_grad=False)
    return states       
  
