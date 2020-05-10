import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.networks import SimpleRNN, TBPTTRNN
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.Indrnn_plainnet import stackedIndRNN_encoder
# -------------------------------------------------------
# Ind RNN 
# -------------------------------------------------------


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.train()   

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'weight_hh' in name:
        param.data.clamp_(-clip,clip)

def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)


class IndRNN(VanillaRNN):
    def __init__(self, opt):
        super(IndRNN, self).__init__(opt)
        self.gradientclip_value=opt.gradclipvalue

        if opt.U_bound==0:
            self.U_bound=np.power(10,(np.log10(opt.MAG)/opt.seq_len))   
        else:
            self.U_bound=opt.U_bound

    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        # self.rnn_model = stackedIndRNN_encoder(self.opt, self.input_size, self.output_size)
 
        self.rnn_model = stackedIndRNN_encoder(self.opt, self.input_size, self.output_size, self.U_bound)  
     
        self.rnn_model.cuda()
        #use_weightdecay_nohiddenW:
        self.param_decay=[]
        self.param_nodecay=[]
        for name, param in self.rnn_model.named_parameters():
            if 'weight_hh' in name or 'bias' in name:
                self.param_nodecay.append(param)      
                #print('parameters no weight decay: ',name)          
            else:
                self.param_decay.append(param)      
                #print('parameters with weight decay: ',name) 

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
        self.inputs = self.inputs.permute(1, 0, 2).to(self.device)
        # self.inputs = self.inputs.view(self.seq_len, -1, self.input_size).to(self.device)

        self.labels = self.labels.view(-1).to(self.device)
       
        if self.opt.predic_task == 'Binary':
            self.labels = self.labels.float()
        # update batch 
        self.batch_size = self.labels.shape[0]

   
    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

      
        self.first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        self.last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop
        
        # # self.optimizer.zero_grad()
        self.rnn_model.zero_grad()
        
        if self.opt.constrain_U:
            clip_weight(self.rnn_model, self.U_bound)
        self.outputs=self.rnn_model(self.inputs)
        self.loss = self.criterion(self.outputs, self.labels)
        # # pred = output.data.max(1)[1] # get the index of the max log-probability
        # # accuracy = pred.eq(targets.data).cpu().sum()      
        self.loss.backward()
     
        # print(self.gradientclip_value)
        clip_gradient(self.rnn_model, self.gradientclip_value)
        if 'FGSM' in self.opt.optimizer:
            if self.opt.iterB > 0:
                self.optimizer.step(self.total_batches, sign_option=True)
            else:
                self.optimizer.step(self.total_batches)
        else:
            self.optimizer.step()

        
   
 
   
        # tacc=tacc+accuracy.numpy()/(0.0+targets.size(0))#loss.data.cpu().numpy()#accuracy
        # count+=1
        if self.last_iter:
            # After last iterT, track Delta w, loss and acc
            self.track_grad_flow(self.rnn_model.named_parameters())
            self.losses.append(self.loss.detach().item())
            self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))

    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            outputs=self.rnn_model(self.inputs)
            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))


