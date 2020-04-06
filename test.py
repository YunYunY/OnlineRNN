import torchvision
import torchvision.transforms as transforms
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from parallel import DataParallelModel, DataParallelCriterion
import time
from torch.nn.parallel import DistributedDataParallel as DDP


BATCH_SIZE = 256

torch.manual_seed(10)
# list all transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)




# parameters 
N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 512
N_OUTPUTS = 10
N_EPHOCS = 10

# This will give accuracy about 95 N_NEURONS =150 will give accuracy 96.45
class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, device):
        super(ImageRNN, self).__init__()
        
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = device
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
  
     
    # def init_hidden(self,):
    #     # (num_layers, batch_size, n_neurons)
    #     # return (torch.zeros(1, self.batch_size, self.n_neurons).to(self.device))
    #     return (torch.zeros(self.batch_size, 1, self.n_neurons).to(self.device))

    def forward(self, X):
        # print(f"Model: input size {X.size()}")
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2) 
        
        self.batch_size = X.size(1)
        self.hidden = init_hidden(self.batch_size, self.n_neurons)
        self.hidden = self.hidden.permute(1, 0, 2)
        out, self.hidden = self.basic_rnn(X, self.hidden)    
        out = out[-1]
  
        out = self.FC(out)
        # print("\toutput size", out.size())
        return out.view(-1, self.n_outputs) # batch_size X n_output

def init_hidden(batch_size, n_neurons):
    # (num_layers, batch_size, n_neurons)
    # return (torch.zeros(1, self.batch_size, self.n_neurons).to(self.device))
    return (torch.zeros(batch_size, 1, n_neurons).to(device))

import torch.optim as optim

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instance
model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, device)

# model.cuda()

criterion = nn.CrossEntropyLoss()
ngpus = torch.cuda.device_count()
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)

model.to(device)
model = DataParallelModel(model)             # Encapsulate the model
criterion  = DataParallelCriterion(criterion) # Encapsulate the loss function

# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

epoch_start_time = time.time()  # timer for entire epoch

for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()
    
    # TRAINING ROUND
    for i, data in enumerate(trainloader):
         # zero the parameter gradients
        optimizer.zero_grad()
        
        # reset hidden states
        # model.hidden = model.init_hidden()
     
        
        # get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 28,28).to(device)
        labels = labels.to(device)
        # forward + backward + optimize
        # outputs = model(inputs)
        outputs = model(inputs).to(device)

        loss = criterion(outputs, labels)
        loss.mean().backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, BATCH_SIZE/ngpus)
        # print('i:  %d | Loss: %.4f' 
        #   %(i, loss.detach().item()))
    model.eval()
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' 
          %(epoch, train_running_loss / i, train_acc/i))
print(time.time()-epoch_start_time)