import torch
import math
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------
# Frank-Wolfe algorithm 1
# --------------------------------------------------------
class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self, total_batches, first_chunk=None, last_chunk=None):
        continue_Adam = self.optimizers[0].step(total_batches, first_chunk, last_chunk)
        if continue_Adam:
            self.optimizers[1].step()
        # for op in self.optimizers:
        #     try:
        #         op.step(total_batches, first_chunk, last_chunk, sign_option)
        #     except:
        #         op.step()

 

class FGSM(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        iterT (positive int): inner iteration
        mergeadam (boolean, optional): whether to merge FGSM with Adam (default: False)
    Example:
        >>> from FGSM import *
        >>> optimizer = FGSM(model.parameters(), lr=0.1, iter=3)
        >>> optimizer.step(total_batches)
 
    Reference: 
        https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
        https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        https://medium.com/the-artificial-impostor/sgd-implementation-in-pytorch-4115bcb9f02c
        https://github.com/facebookarchive/adversarial_image_defenses/blob/master/adversarial/lib/adversary.py
    """
    def __init__(self, params, lr=required, iterT=required, mergeadam=False):
        defaults = dict(lr=lr, mergeadam=mergeadam)
        super(FGSM, self).__init__(params, defaults)
        self.iterT = iterT
    

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)
    

    def step(self, total_batches, first_chunk=None, last_chunk=None):
        """ Performs a single optimization step.
        Arguments:
            total_batches: total batch number 
            first_chunk: is first_iter when not in tbptt, otherwise it's first chunk in inner iteration 
            last_chunk: similar to first_chunk, for stand for last one
        """
        # Adam update only works in the last step after inner iteration
        continue_Adam = False

        first_iter = (total_batches-1)%self.iterT == 0 # if this is the first iter of inner loop
        last_iter = (total_batches-1)%self.iterT == (self.iterT-1) # if this is the last step of inner loop
        
        t = (total_batches-1)%self.iterT + 1

        if not first_chunk and not last_chunk:
            first_chunk = first_iter
            last_chunk = last_iter
        
     
        for group in self.param_groups:

            lr = group['lr']

            for p in group['params']:
              
                if first_chunk:
                    p.data_orig = p.data.clone() # keep original weights
                param_state = self.state[p]

                #----------------------FGSM inner loop-----------------------------------------------
                if first_chunk:
                # if first_iter:
                    # initialize Delta g0 as 0, param_state['momentum_buffer'] will change according to buf
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                  
                buf = param_state['momentum_buffer']
             
                # -------------------FGSM with grad value ---------------------    
                
                d_p = p.grad.data
                d_p_norm = torch.norm(d_p)
                d_p.div_(d_p_norm)
          

                buf.mul_(1.-1./t).add_(-lr/t, d_p) # update Delta g_k
                # buf.mul_(1.-2./(t+2.)).add_(-2.*lr/(t+2.), d_p) 
                
                p.data.add_(buf) # update weights w_k = Delta w_k-1 + Delta g_k
                p.grad = buf.clone()    
                
                # ------------------- ----------last iterT update w-------------------------
                if last_chunk:
                    
                    with torch.no_grad():
                        # recover w_k to original value before the last iterT of inner loop 
                        p.data = p.data_orig.clone()

                    p.grad.div_((-1))
                    # p.grad.div_((-lr)) # rescale before feeding into Adam

                    # set to True so that second optimizer can work
                    continue_Adam = True

        return continue_Adam 



