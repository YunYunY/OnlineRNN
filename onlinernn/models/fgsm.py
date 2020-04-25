import torch
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np
class FGSM(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        iterT (float, optional): ratio of long to short step (default: 1)
        mu (float, optional): momentum coefficient (default: 0.99)
    Example:
        >>> from FGSM import *
        >>> optimizer = FGSM(model.parameters(), lr=0.1, mu = 0.99, iterT = 1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    Reference: 
        https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
        https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        https://medium.com/the-artificial-impostor/sgd-implementation-in-pytorch-4115bcb9f02c
        https://github.com/facebookarchive/adversarial_image_defenses/blob/master/adversarial/lib/adversary.py
    """
    def __init__(self, params, lr=required, mu = 0.9, iterT = 1, weight_decay=0):
        defaults = dict(lr=lr, mu=mu, iterT=iterT, weight_decay=weight_decay)
        super(FGSM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            mu = group['mu']
            iterT = group['iterT']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_sgin = p.grad.sign()

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    # initialize delta omega0 as 0
                    # TODO try initialization like pytorch current official code
                    # TODO check MIT thesis defense paper 3
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = param_state['momentum_buffer']
                # inside iteration to update velocity
                for i in range(iterT):
                    buf.mul_(mu)
                    buf.add_(-lr,grad_sgin)
             
                p.data.add_(lr,buf)
        
        return loss



