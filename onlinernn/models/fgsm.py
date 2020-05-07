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

    def step(self, total_batches, first_chunk=None, last_chunk=None, sign_option=None):
        for op in self.optimizers:
            try:
                op.step(total_batches, first_chunk, last_chunk, sign_option)
            except:
                op.step()

 

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
        >>> optimizer.step()
 
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
        self.inside_loop = True # use inner loop or not, default True
        self.sign_vote = True # let all previous sign vote for the last iter's update, default False
        self.fgsm_apply_sign = True # in FGSM method, apply sign or magnitude, default True  

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)
    

    def step(self, total_batches, first_chunk=None, last_chunk=None, sign_option=None):
        """ Performs a single optimization step.
        Arguments:
            total_batches: total batch number 
            first_chunk: is first_iter when not in tbptt, otherwise it's first chunk in inner iteration 
            sign_option: apply sign option method modified from Algorithm3 in https://arxiv.org/pdf/1802.04434.pdf
        """
        loss = None
        if total_batches == 1:
            print(f'inner loop is {self.inside_loop}')
            print(f'sign average is {sign_option}')
        if self.inside_loop:
            # inner loop
            first_iter = (total_batches-1)%self.iterT == 0 # if this is the first iter of inner loop
            last_iter = (total_batches-1)%self.iterT == (self.iterT-1) # if this is the last step of inner loop
           
            t = (total_batches-1)%self.iterT + 1
            # --------------------Case for TBPTT---------------------------
            if not first_chunk and not last_chunk:
                first_chunk = first_iter
                last_chunk = last_iter
        else:
            first_iter = True 
            last_iter = True 
            t = total_batches
        
     
        for group in self.param_groups:

            lr = group['lr']
            mergeadam = group['mergeadam']

            # weight_decay = group['weight_decay']

            for p in group['params']:
               
                if first_chunk:
                    p.data_orig = p.data.clone() # keep original weights
                param_state = self.state[p]

                #----------------------FGSM inner loop---------------------------------------------------------
                if self.inside_loop:
                    if first_iter:
                        # initialize Delta w0 as 0, param_state['momentum_buffer'] will change according to buf
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        # --------------------------multiple sign vote method-----------------------------------
                        if self.sign_vote and first_chunk:
                            param_state['sign_vote'] = torch.zeros_like(p.data)

                    buf = param_state['momentum_buffer']
                    grad_sign = p.grad.sign()
                    # -------------------------------Modified Algorithm 3 from Amazon sign paper------------------
                    if sign_option:
                        # Modified average sign method
                        buf.add_(grad_sign)
                    else:
                        #--------------------------FGSM----------------------------------
                        if self.fgsm_apply_sign:
                            # -------------------FGSM with sign of grad------------------
                            buf.mul_(1.-1./t).add_(-lr/t, grad_sign) # update Delta w_t 

                        else: 
                            # -------------------FGSM with grad value---------------------
                            buf.mul_(1.-1./t).add_(-lr/t, p.grad) # update Delta w_t 
                    # ------------------------Algorithm 3 in Amazon paper------------------
                    if self.sign_vote:
                        param_state['sign_vote'] += buf.sign()
                else:
                    # --------------------------------No inner loop-------------------------------------------
                    # Momentum FGSM
                    if 'momentum_buffer' not in param_state:
                        # initialize Delta w0 as 0, param_state['momentum_buffer'] will change according to buf
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)

                    buf = param_state['momentum_buffer']
                    grad_sign = p.grad.sign()
                    mu1 = 1.-1/t
                    mu2 = -lr/t
                    buf.mul_(mu1).add_(mu2, grad_sign) # update Delta w_t 
                    # buf.mul_(1.-2./(t+1)).add_(-2*lr/(t+1), grad_sign) # update Delta w_t 
                # -------------------------------------modified Algorithm 3 of Amazon paper------------
                if sign_option is None:
                    p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                    
                p.buf = buf.clone() 
          
                # -----------------------------last iterT update w-------------------------
                if self.inside_loop and last_chunk:
                    with torch.no_grad():
                        # recover w_k to original value before inner loop at the last iterT
                        p.data = p.data_orig.clone()
                    # ----------------------------Adam method-------------------------------
                    if mergeadam:
                        # use Delta w_t in adam as grad, need to rescale
                        if self.sign_vote:
                            p.grad = param_state['sign_vote'].sign()
                        else:
                            p.grad = buf.clone()/(-lr)
                    # ---------------------------modified Algorithm3 in exp 10--------------
                    elif sign_option:
                        p.data.add_(-lr, buf.sign())
                    # ----------------------------general FGSM-------------------------------
                    else:
                        # update w_k by adding Detal w_t 
                        p.data.add_(p.buf)
                                
        return loss



