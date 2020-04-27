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
    Example:
        >>> from FGSM import *
        >>> optimizer = FGSM(model.parameters(), lr=0.1, iterT = 1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    Reference: 
        https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
        https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        https://medium.com/the-artificial-impostor/sgd-implementation-in-pytorch-4115bcb9f02c
        https://github.com/facebookarchive/adversarial_image_defenses/blob/master/adversarial/lib/adversary.py
    """
    def __init__(self, params, lr=required, iterT = 1):
        defaults = dict(lr=lr, iterT=iterT)
        super(FGSM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FGSM, self).__setstate__(state)

    def step(self, forward, criterion, labels, device):
        """ Performs a single optimization step.
        Arguments:
            closure_forward (callable) forward
            criterion: loss function 
            labels: target labels 
        """

        outputs, _ = forward()
        outputs = outputs.to(device)
    
        # calculate loss
        orig_loss = criterion(outputs, labels)

        # calculate w_k     
        orig_loss.backward(retain_graph = True)

        for group in self.param_groups:

            lr = group['lr']
            iterT = group['iterT']

            for p in group['params']:
                if p.grad is None:
                    continue
                p_data_orig = p.data # keep original weights

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    # initialize Delta w0 as 0
                    # TODO try initialization like pytorch current official code
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = param_state['momentum_buffer']

                # inside iteration to update velocity
                for t in range(1, iterT+1):
                    p.grad.data.zero_() # zero gradient to prevent grad accumulation
                    p.data.add_(buf) # update weights w_t = w_k + Delta w_(t-1)
                    outputs, _ = forward() # forward after w changes 
                    outputs = outputs.to(device)
                    loss = criterion(outputs, labels) # calculate loss 
                    loss.backward(retain_graph = True) # calculate gradient wrt Delta w
                    grad_sign = p.grad.sign()
                    buf.mul_(1.-1./t).add_(-lr/t, grad_sign) # update Delta w_t 
          
                # recover weights to original 
                with torch.no_grad():
                    p.data = p_data_orig
                p.buf = buf

            # update parameters when all bufs are calculated
            for p in group['params']:
                p.data.add_(p.buf)
    
        return orig_loss, outputs



