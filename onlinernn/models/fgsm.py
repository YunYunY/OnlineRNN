import torch
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np
import warnings
warnings.filterwarnings('ignore')
class FGSM(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        iterT (positive int): inner iteration
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
    def __init__(self, params, lr=required, iterT=required):
        defaults = dict(lr=lr)
        super(FGSM, self).__init__(params, defaults)
        self.iterT = iterT

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)

    def step(self, total_batches):
        """ Performs a single optimization step.
        Arguments:
            total_batches: total batch number 
        """
        # print(self.state)
        loss = None

        first_iter = (total_batches-1)%self.iterT == 0 # if this is the first iter of inner loop
        last_iter = (total_batches-1)%self.iterT == (self.iterT-1) # if this is the last step of inner loop
        t = (total_batches-1)%self.iterT + 1

        for group in self.param_groups:

            lr = group['lr']
            # weight_decay = group['weight_decay']

            for p in group['params']:
                # d_p = p.grad
                # if weight_decay != 0:
                #     d_p = d_p.add(p, alpha=weight_decay)

                if first_iter:
                    p.data_orig = p.data.clone() # keep original weights
                param_state = self.state[p]
                # if 'momentum_buffer' not in param_state:
                if first_iter:
                    # initialize Delta w0 as 0, param_state['momentum_buffer'] will change according to buf
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = param_state['momentum_buffer']

                grad_sign = p.grad.sign()
                buf.mul_(1.-1./t).add_(-lr/t, grad_sign) # update Delta w_t 
                # print(p.data)
                # print(buf)
                p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                # print(p.data)
             
                # recover weights to original at the last iterT 
                if last_iter:
                    with torch.no_grad():
                        p.data = p.data_orig.clone()
                        p.buf = buf.clone() 
                        p.data.add_(p.buf)

        return loss



