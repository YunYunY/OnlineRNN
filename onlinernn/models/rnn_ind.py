import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from onlinernn.models.rnn_tbptt import TBPTT
from onlinernn.models.rnn_vanilla import VanillaRNN
from onlinernn.models.Indrnn_plainnet import stackedIndRNN_encoder
from onlinernn.models.indrnn_utils import set_bn_train, clip_weight, clip_gradient, generate_subbatches
# -------------------------------------------------------
# Ind RNN 
# -------------------------------------------------------


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
        if self.dataname in ['ADDING']:
            print('here')
            exit(0)
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
   
        self.labels = self.labels.view(-1).to(self.device)
       
        if self.opt.predic_task == 'Binary':
            self.labels = self.labels.float()
        # update batch 
        self.batch_size = self.labels.shape[0]

    def init_states(self):
        self.states = torch.zeros((self.opt.num_layers, self.batch_size,self.opt.hidden_size)).to(self.device)

    def train_subsequence(self):

        self.init_states()
    
        nchunks = self.seq_len // self.opt.subseq_size

        # for i in range(self.opt.iterT):
        for i in range(nchunks):

            sub_inputs  = generate_subbatches(self.inputs, i, size=self.opt.subseq_size)
           
            self.rnn_model.zero_grad()
            self.states = self.states.detach()
            if self.opt.constrain_U:
                clip_weight(self.rnn_model, self.U_bound)
            self.outputs, self.states =self.rnn_model(sub_inputs, self.states)
        
            self.loss = self.criterion(self.outputs, self.labels)
            self.loss.backward()

            clip_gradient(self.rnn_model, self.gradientclip_value)

            if 'FGSM' in self.opt.optimizer:

                tbptt_first_iter = self.first_iter and i==0 # first iterT and first chunk in tbptt
                tbptt_last_iter = self.last_iter and i==nchunks-1 # last iterT and last chunk in tbptt
                # tbptt_first_iter = True
                # tbptt_last_iter = True
                # tbptt_first_iter = i==0 # first iterT and first chunk in tbptt
                # tbptt_last_iter = i==self.opt.iterT-1 # last iterT and last chunk in tbptt
               
                self.optimizer.step(self.total_batches, tbptt_first_iter, tbptt_last_iter)
            else:
                self.optimizer.step()
       
    def forward_indrnn(self):
        self.rnn_model.zero_grad()
        if self.opt.constrain_U:
            clip_weight(self.rnn_model, self.U_bound)
        self.outputs =self.rnn_model(self.inputs)

    def backward_indrnn(self):
        self.loss = self.criterion(self.outputs, self.labels)
    
        self.loss.backward()
    
        clip_gradient(self.rnn_model, self.gradientclip_value)
        if 'FGSM' in self.opt.optimizer:
            # if self.opt.iterB > 0:
            #     self.optimizer.step(self.total_batches, sign_option=True)
            # else:
            self.optimizer.step(self.total_batches)
        else:
            self.optimizer.step()


    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        self.last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop
        
        if self.opt.subsequene:
            # tbptt train, tbptt segments and iterT is same. 
            self.train_subsequence()     
            # self.track_grad_flow(self.rnn_model.named_parameters())
            # self.losses.append(self.loss.detach().item())
            # self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))         
        else:
            self.forward_indrnn()
            self.backward_indrnn()
       
        if self.last_iter:
            # after last iterT, track Delta w, loss and acc
            # self.track_grad_flow(self.rnn_model.named_parameters())
            self.losses.append(self.loss.detach().item())
            self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))

    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            if self.opt.subsequene:
                self.init_states()

                outputs, _ =self.rnn_model(self.inputs, self.states)
            else:
                outputs =self.rnn_model(self.inputs)

            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))


