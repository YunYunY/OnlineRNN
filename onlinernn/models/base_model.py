from abc import ABC
import numpy as np
import torch
import copy
# from onlinernn.models.fgsm_sign import FGSM, MultipleOptimizer
from onlinernn.models.fgsm import FGSM, MultipleOptimizer
# from onlinernn.models.fgsm_fw_2 import FGSM, MultipleOptimizer
from onlinernn.models.svrg import SVRG_k, SVRG_Snapshot
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
        try:
            self.rad = opt.rad
        except:
            pass
        self.seq_len = opt.seq_len
        self.num_layers = opt.num_layers
        self.init_mode = opt.init_mode
        self.hidden_size = opt.hidden_size
        self.input_size = opt.feature_shape
        self.output_size = opt.n_class
        self.T = opt.iterT
        self.gradientclip_value = opt.gradclipvalue
        self.device = opt.device
    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
        if self.opt.LSTM:
            return 'LSTM'
        return self.class_name()

    # ----------------------------------------------
    def init_optimizer(self):
        """
        Setup optimizers
        """
        self.optimizers = []
        if self.opt.optimizer == "SVRG":
            self.optimizer = SVRG_k(self.rnn_model.parameters(), lr=self.lr, weight_decay=0)
            self.optimizer_snapshot = SVRG_Snapshot(self.model_snapshot.parameters())
        elif self.opt.optimizer == 'Adam':
            self.optimizer = Adam(self.rnn_model.parameters(), lr=self.lr)
        elif self.opt.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr=self.lr)
        elif self.opt.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr)
        elif self.opt.optimizer == 'SGD_Momentum':
            self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt.optimizer == 'irnn_Adam':
            self.optimizer = torch.optim.Adam([
                    {'params': self.param_nodecay},
                    {'params': self.param_decay, 'weight_decay': self.opt.decayfactor}
                ], lr=self.lr) 
        elif self.opt.optimizer == 'FGSM':
            self.optimizer = FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T)
        elif self.opt.optimizer == 'FGSM_SGD':
            self.optimizers = [FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T, mergeadam=True)] + \
                        [torch.optim.SGD(self.rnn_model.parameters(), lr=self.lr)]    
        elif self.opt.optimizer == 'FGSM_Adam':
            self.optimizers = [FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T, mergeadam=True)] + \
                        [Adam(self.rnn_model.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)]     
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
        if self.opt.single_output:
            return self.inputs[:, i*size: (i+1)*size, :], self.labels
        else:
            return self.inputs[:, i*size: (i+1)*size, :], self.labels[:, i*size: (i+1)*size, :]

    def global_train(self):
        self.optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
        for i, data in enumerate(self.dataset.dataloader):
            self.data = data 
            self.set_input()
            self.init_states()
            outputs, _ = self.model_snapshot(self.inputs, self.states)
            self.outputs = outputs.to(self.device)     
            snapshot_loss = self.criterion(self.outputs, self.labels) 
            snapshot_loss.backward()
        
        # pass the current paramesters of optimizer_0 to optimizer_k 
        u = self.optimizer_snapshot.get_param_groups()
        self.optimizer.set_u(u)
    
        for i, data in enumerate(self.dataset.dataloader):
            self.data = data 
            self.set_input()
            self.optimizer.zero_grad()
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states)
            self.outputs = outputs.to(self.device)
            self.loss = self.criterion(self.outputs, self.labels)   
            self.loss.backward()

        #     # optimization 
        #     loss_iter.backward()    

        #     yhat2 = model_snapshot(images)
        #     loss2 = loss_fn(yhat2, labels)

        #     optimizer_snapshot.zero_grad()
        #     loss2.backward()

        #     optimizer_k.step(optimizer_snapshot.get_param_groups())

        #     # logging 
        #     acc_iter = accuracy(yhat, labels)
        #     loss.update(loss_iter.data.item())
        #     acc.update(acc_iter)
        
        # # update the snapshot 
        # optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
        
        # return loss.avg, acc.avg
   



    
    def partial_grad(self, model):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs, _ = model(self.inputs, self.states)
        self.outputs = outputs.to(self.device)        
        loss = self.criterion(self.outputs, self.labels) 
        loss.backward()
        return loss
    '''
    # ----------------------------------------------
    def calculate_loss_grad(self):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0
        full_grad = 0.0
        for i_grad, data_grad in enumerate(self.dataset.dataloader):
            self.data = data_grad 
            self.set_input()
            self.init_states()
            total_loss += (1./self.batch_size ) * self.partial_grad(self.previous_net_grad)
        
        for para in self.previous_net_grad.parameters():
            if para.grad is None:
                continue 
            full_grad += para.grad.data.norm(2)**2
        return total_loss, (1./self.batch_size) * torch.sqrt(full_grad)

    # ----------------------------------------------
    def calculate_global_grad(self):

        self.previous_net_sgd = copy.deepcopy(self.rnn_model) #update previous_net_sgd
        self.previous_net_grad = copy.deepcopy(self.rnn_model) #update previous_net_grad
        self.previous_net_grad.zero_grad() # grad = 0

        self.total_loss_epoch, self.grad_norm_epoch = self.calculate_loss_grad()


    def calculate_previous_grad(self):
        #Compute prev stoc grad
        self.previous_net_sgd.zero_grad() #grad = 0
        prev_loss = self.partial_grad(self.previous_net_sgd)

    '''
 