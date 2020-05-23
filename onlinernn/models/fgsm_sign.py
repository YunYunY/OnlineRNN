import torch
import math
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
        #         op.step(total_batches, first_chunk, last_chunk)
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
        self.sign_vote = True # let all previous sign vote for the last iter of inner loop's update, default False
        self.fgsm_apply_sign = False # in FGSM method, apply sign or magnitude, default True  
        self.last_step_norm = True
    

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

        # --------------------for TBPTT---------------------------
        if not first_chunk and not last_chunk:
            first_chunk = first_iter
            last_chunk = last_iter
   
    
     
        for group in self.param_groups:

            lr = group['lr']
            mergeadam = group['mergeadam']

            for p in group['params']:
              
                if first_chunk:
                    p.data_orig = p.data.clone() # keep original weights
                param_state = self.state[p]

                #----------------------FGSM inner loop---------------------------------------------------------
                if first_iter:
                    # initialize Delta w0 as 0, param_state['momentum_buffer'] will change according to buf
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    # --------------------------multiple sign vote method-----------------------------------
                    if self.sign_vote and first_chunk:
                        param_state['sign_vote'] = torch.zeros_like(p.data)
                        param_state['n_update']  = 0.

                buf = param_state['momentum_buffer']
                grad_sign = p.grad.sign()
            
                
                #--------------------------FGSM----------------------------------
                if self.fgsm_apply_sign:
                    # -------------------FGSM with sign of grad------------------
                    buf.mul_(1.-1./t).add_(-lr/t, grad_sign) # update Delta w_t 
                    # buf.mul_(1.-2./(t+2.)).add_(-2.*lr/(t+2.), grad_sign) # update Delta w_t 

                else: 
                    # -------------------FGSM with grad value ---------------------    
                    # use grad only
                    buf.mul_(1.-1./t).add_(-lr/t, p.grad) # update Delta w_t 
                    # buf.mul_(1.-2./(t+2.)).add_(-2.*lr/(t+2.), p.grad) # update Delta w_t 

                # ------------------------Algorithm 3 in Amazon paper exp 10------------------
                if self.sign_vote:
                    param_state['sign_vote'] += buf.sign()
                    param_state['n_update'] += 1.
            

                # -------------------------------------update grad and p.data--------------------------    
                # # ----------------------------Adam method-------------------------------
                # if mergeadam:
                #     p.grad = buf.clone()/(-lr)
                #     # use Delta w_t in adam as grad, need to rescale
                # else:
                #     p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                #     p.grad = buf.clone()
                
                p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                p.grad = buf.clone()    
                
                
                
                # ------------------- ----------last iterT update w-------------------------
                if last_chunk:
                    
                    with torch.no_grad():
                        # recover w_k to original value before the last iterT of inner loop 
                        p.data = p.data_orig.clone()
              
                    if mergeadam:
                        if self.sign_vote:
                            # average of sum of sign
                            p.grad = -param_state['sign_vote']/param_state['n_update']
                            # sign of sum of sign
                            # p.grad = -param_state['sign_vote'].sign()
                        else:
                            p.grad.div_((-lr)) # rescale before feeding into Adam
                            # p.grad = buf.clone()/(-lr)

                        if self.last_step_norm:                                
                            p.grad.div_(torch.norm(p.grad.data))

                        # set to True so that second optimizer can work
                        continue_Adam = True
                    # ----------------------------general FGSM-------------------------------
                    else:
                        # update w_k by adding Detal w_t 
                        p.data.add_(buf.clone())

        return continue_Adam 



