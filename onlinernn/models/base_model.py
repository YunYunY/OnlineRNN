from abc import ABC
import numpy as np
import torch
from onlinernn.models.fgsm import FGSM, MultipleOptimizer
# from onlinernn.models.fgsm_fw_2 import FGSM, MultipleOptimizer

from onlinernn.models.adam import Adam 
# ----------------------------------------------
"""
    Abstract class defining the APIs for model classes
"""
# ----------------------------------------------
class BaseModel(ABC):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.istrain = opt.istrain
        self.lr = opt.lr
        self.seq_len = opt.seq_len
        self.num_layers = opt.num_layers
        self.init_mode = opt.init_mode
        self.hidden_size = opt.hidden_size
        self.input_size = opt.feature_shape
        self.output_size = opt.n_class
        self.T = opt.iterT
        self.device = opt.device
    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
        return self.class_name()

    # ----------------------------------------------
    def init_optimizer(self):
        """
        Setup optimizers
        """
        self.optimizers = []
        # self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        if self.opt.optimizer == 'Adam':
            self.optimizer = Adam(self.rnn_model.parameters(), lr=self.lr)
        elif self.opt.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr=self.lr)
        elif self.opt.optimizer == 'SGD_Momentum':
            self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt.optimizer == 'irnn_Adam':
            self.optimizer = torch.optim.Adam([
                    {'params': self.param_nodecay},
                    {'params': self.param_decay, 'weight_decay': self.opt.decayfactor}
                ], lr=self.lr) 
        elif self.opt.optimizer == 'FGSM':
            self.optimizer = FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T)
        elif self.opt.optimizer == 'FGSM_Adam':
            self.optimizers = [FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T, mergeadam=True)] + \
                        [Adam(self.rnn_model.parameters(), lr=self.lr)]     
        elif self.opt.optimizer == 'FGSM_RMSProp':  
            self.optimizers =  [FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T, mergeadam=True)] + \
                        [torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr)]
        elif self.opt.optimizer == 'FGSM_Adagrad':  
            self.optimizers =  [FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T, mergeadam=True)] + \
                        [torch.optim.Adagrad(self.rnn_model.parameters(), lr=self.lr)]
        
        if len(self.optimizers) == 0:
            self.optimizers.append(self.optimizer)
        else:
            self.optimizer = MultipleOptimizer(*self.optimizers)

 
 
    def init_loss(self):
        """
        Define Loss functions
        """
        if self.loss_method == "MSE":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_method == "CrossEntropy":
            self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)
        elif self.loss_method == "BCELogit":
            self.criterion = torch.nn.BCEWithLogitsLoss()

    # ----------------------------------------------
    def generate_subbatches(self, i, size):
        """
        Generate sub batches of data by splitting data along time sequence
        """
        # for i in range(self.seq_len // size):
        #     yield self.inputs[:, i*size: (i+1)*size, :], self.labels
        return self.inputs[:, i*size: (i+1)*size, :], self.labels

    


 