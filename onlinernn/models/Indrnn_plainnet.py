from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np
from onlinernn.models.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
#if no cuda, then use the following line
#from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from onlinernn.models.indrnn_utils import Batch_norm_overtime,Linear_overtime_module,Dropout_overtime, generate_subbatches, clip_weight, clip_gradient
BN=Batch_norm_overtime
Linear_overtime=Linear_overtime_module
dropout_overtime=Dropout_overtime.apply


class IndRNNwithBN(nn.Sequential):
    def __init__(self, args, hidden_size, seq_len,bn_location='bn_before'):
        super(IndRNNwithBN, self).__init__()  
        if bn_location=="bn_before":      
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        self.add_module('indrnn1', IndRNN(hidden_size))      

        if bn_location=="bn_after":   
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        if (bn_location!='bn_before') and (bn_location!='bn_after'):
            print('Please select a batch normalization mode.')
            assert 2==3

class stackedIndRNN_encoder(nn.Module):
    """
    Define stacked IndRNN architecture
    """
    def __init__(self, args, input_size, outputclass, U_bound):
      
        super(stackedIndRNN_encoder, self).__init__()     
        self.args = args
        self.U_bound = U_bound
        self.U_lowbound=np.power(10,(np.log10(1.0/args.MAG)/args.seq_len))  

        hidden_size=args.hidden_size
      
      
        self.DIs=nn.ModuleList()
        denseinput=Linear_overtime(input_size, hidden_size)
        self.DIs.append(denseinput)
        for x in range(args.num_layers - 1):
            denseinput = Linear_overtime(hidden_size, hidden_size)
            self.DIs.append(denseinput) 

        self.RNNs = nn.ModuleList()
        for x in range(args.num_layers):
            rnn = IndRNNwithBN(args, hidden_size=hidden_size, seq_len=args.seq_len,bn_location=args.bn_location) #IndRNN
            self.RNNs.append(rnn)         
            
        self.classifier = nn.Linear(hidden_size, outputclass, bias=True)

        self.init_weights()

    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0, self.U_bound)          
        if self.args.u_lastlayer_ini and 'RNNs.'+str(self.args.num_layers-1)+'.weight_hh' in name:
          param.data.uniform_(self.U_lowbound, self.U_bound)    
        if ('fc' in name) and 'weight' in name:#'denselayer' in name and 
            nn.init.kaiming_uniform_(param, a=8, mode='fan_in')#
        if 'classifier' in name and 'weight' in name:
            nn.init.kaiming_normal_(param.data)
        if ('norm' in name or 'Norm' in name)  and 'weight' in name:
            param.data.fill_(1)
        if 'bias' in name:
            param.data.fill_(0.0)

    

    def forward(self, input, h0=None):
      
        
        rnnoutputs={}    
        rnnoutputs['outlayer-1']=input
        if self.args.subsequene:
            nextstates_size = list(h0.size())
            nextstates = torch.empty(*nextstates_size, dtype=h0.dtype, device=h0.device)
            for x in range(len(self.RNNs)):
                # fc layer convert [784, 32, 1] -> [784, 32, 128]
                rnnoutputs['dilayer%d'%x]=self.DIs[x](rnnoutputs['outlayer%d'%(x-1)])    
                # indRNN layer [784, 32, 128] -> [784, 32, 128]
                input_dic = {}
                input_dic[0] = rnnoutputs['dilayer%d'%x]
                input_dic[1] = h0[x]
                rnnoutputs['outlayer%d'%x]= self.RNNs[x](input_dic)   

                if self.args.dropout>0:
                    rnnoutputs['outlayer%d'%x]= dropout_overtime(rnnoutputs['outlayer%d'%x],self.args.dropout,self.training) 
                nextstates[x] = rnnoutputs['outlayer%d'%x][-1]
            # [seq_len, 32, 128] -> [32, 128]
            temp=rnnoutputs['outlayer%d'%(len(self.RNNs)-1)][-1]
            output = self.classifier(temp)

            return output, nextstates     
     
       
        else:
            for x in range(len(self.RNNs)):
                # fc layer convert [seq_len, 32, 1] -> [seq_len, 32, 128]
                rnnoutputs['dilayer%d'%x]=self.DIs[x](rnnoutputs['outlayer%d'%(x-1)])    
                # indRNN layer [seq_len, 32, 128] -> [seq_len, 32, 128]
                rnnoutputs['outlayer%d'%x]= self.RNNs[x](rnnoutputs['dilayer%d'%x])   
                if self.args.dropout>0:
                    rnnoutputs['outlayer%d'%x]= dropout_overtime(rnnoutputs['outlayer%d'%x],self.args.dropout,self.training) 
            # [seq_len, 32, 128] -> [32, 128]
            temp=rnnoutputs['outlayer%d'%(len(self.RNNs)-1)][-1]
            output = self.classifier(temp)

            # hidden = rnnoutputs['outlayer%d'%(len(self.RNNs)-1)]

            return output                
        
      
