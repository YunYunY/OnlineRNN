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

    def step(self, total_batches):
        for op in self.optimizers:
            try:
                op.step(total_batches)
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
        self.inside_loop = True
        self.sign_vote = False # use Algorithm3 in https://arxiv.org/pdf/1802.04434.pdf

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)
    


    def step(self, total_batches, sign_option=None):
        """ Performs a single optimization step.
        Arguments:
            total_batches: total batch number 
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
         
        else:
            first_iter = True 
            last_iter = True 
            t = total_batches
        
     
        for group in self.param_groups:

            lr = group['lr']
            mergeadam = group['mergeadam']

            # weight_decay = group['weight_decay']

            for p in group['params']:
                # d_p = p.grad
                # if weight_decay != 0:
                #     d_p = d_p.add(p, alpha=weight_decay)

                if first_iter:
                    p.data_orig = p.data.clone() # keep original weights
                param_state = self.state[p]

                if self.inside_loop:
                    if first_iter:
                        # initialize Delta w0 as 0, param_state['momentum_buffer'] will change according to buf
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        if self.sign_vote:
                            # Vote method 
                            param_state['sign_vote'] = torch.zeros_like(p.data)

                    buf = param_state['momentum_buffer']
                    grad_sign = p.grad.sign()
                    if sign_option:
                        # Modified average sign method
                        buf.add_(grad_sign)
                    else:
                        buf.mul_(1.-1./t).add_(-lr/t, grad_sign) # update Delta w_t 
             
                    if self.sign_vote:
                        param_state['sign_vote'] += buf.sign()
                else:
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
                if sign_option is None:
                    p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                    
                p.buf = buf.clone() 
                if self.inside_loop and last_iter:
                    with torch.no_grad():
                        # recover w_k to original value before inner loop at the last iterT
                        p.data = p.data_orig.clone()
                        # p.buf = buf.clone() 
                    if mergeadam:
                        # use Delta w_t in adam 
                        p.grad = buf.clone()/(-lr)
                    elif self.sign_vote:
                        p.data.add_(-lr, param_state['sign_vote'].sign())
                    elif sign_option:
                        p.data.add_(-lr, buf.sign())
                    else:
                        # update w_k by adding Detal w_t 
                        p.data.add_(p.buf)
                                
        return loss



