"""
This code is to implement the IndRNN (only the recurrent part) using CUDA for fast computation. The CUDA part is similar the SRU implementation from 
https://github.com/taolei87/sru.
This runs around 32 times faster than the general pytorch implementation on pixel MNIST example (sequence lengeth 784). For longer sequence, 
it will be even more efficient, and vice versa. 
Since this only contains the recurrent part of IndRNN, fully connected layers or convolutional layers are needed before it.
Please cite the following paper if you find it useful.
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN," 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5457-5466. 2018.
@inproceedings{li2018independently,
  title={Independently recurrent neural network (indrnn): Building A longer and deeper RNN},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5457--5466},
  year={2018}
}
"""

import sys
import time
import math
#import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from torch.nn import Parameter
#import torch.nn.functional as F

from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

IndRNN_CODE = """
extern "C" {

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __forceinline__ __device__ float calc_grad_activation(float x)
    {
        return (x > 0.f) ? 1.f : 0.f;
    }

    __global__ void indrnn_fwd( const float * __restrict__ x,
                            const float * __restrict__ weight_hh, const float * __restrict__ h0,
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ h)
    {
        int ncols = batch*hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;       
        const float weight_hh_cur = *(weight_hh + (col%hidden_size));
        float cur = *(h0 + col);
        const float *xp = x+col;
        float *hp = h+col;

        for (int row = 0; row < len; ++row)
        {
            cur=reluf(cur*weight_hh_cur+(*xp));
            *hp=cur;
            xp += ncols;
            hp += ncols;            
        }
    }

    __global__ void indrnn_bwd(const float * __restrict__ x,
                             const float * __restrict__ weight_hh, const float * __restrict__ h0,
                             const float * __restrict__ h,
                            const float * __restrict__ grad_h, 
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_weight_hh, float * __restrict__ grad_h0)
    {    
        int ncols = batch*hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;        
        const float weight_hh_cur = *(weight_hh + (col%hidden_size));
        float gweight_hh_cur = 0;
        float cur = 0;  // *(grad_last + col);        //0; strange gradient behavior. grad_last and grad_h, one of them is zero.     
        
        const float *xp = x+col + (len-1)*ncols;
        const float *hp = h+col + (len-1)*ncols;      
        float *gxp = grad_x + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        

        for (int row = len-1; row >= 0; --row)
        {        
            const float prev_h_val = (row>0) ? (*(hp-ncols)) : (*(h0+col));
            //float h_val_beforeact = prev_h_val*weight_hh_cur+(*xp);
            float gh_beforeact = ((*ghp) + cur)*calc_grad_activation(prev_h_val*weight_hh_cur+(*xp));
            cur = gh_beforeact*weight_hh_cur;
            gweight_hh_cur += gh_beforeact*prev_h_val;
            *gxp = gh_beforeact;

            xp -= ncols;
            hp -= ncols;
            gxp -= ncols;
            ghp -= ncols;        
        }

        atomicAdd(grad_weight_hh + (col%hidden_size), gweight_hh_cur);
        *(grad_h0 +col) = cur;
    }
}
"""

class GetFunctions():
    _IndRNN_PROG = Program(IndRNN_CODE, 'indrnn_prog.cu')#.encode('utf-8')  .encode()
    _IndRNN_PTX = _IndRNN_PROG.compile()
    _DEVICE2FUNC = {}

    def compile_functions(self):
        device = torch.cuda.current_device()
        print ('IndRNN loaded for gpu {}'.format(device))
        mod = function.Module()
        mod.load(bytes(self._IndRNN_PTX.encode()))
        fwd_func = mod.get_function('indrnn_fwd')
        bwd_func = mod.get_function('indrnn_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (current_stream, fwd_func, bwd_func)
        return current_stream, fwd_func, bwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

class IndRNN_Compute_GPU(Function):

    def __init__(self):
        super(IndRNN_Compute_GPU, self).__init__()

    @staticmethod
    def forward(ctx, x, weight_hh, h0):
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        hidden_size = x.size(-1)  #hidden_size
        ncols = batch*hidden_size
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1
        
        size = (length, batch, hidden_size) if x.dim() == 3 else (batch, hidden_size)
        h = x.new(*size)
        stream, fwd_func, _ = GetFunctions().get_functions()

        # stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            x.contiguous().data_ptr(),
            weight_hh.contiguous().data_ptr(),
            h0.contiguous().data_ptr(),
            length,
            batch,
            hidden_size,
            h.contiguous().data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        # self.save_for_backward(x, h, weight_hh, h0)#
        # gradclipvalue = torch.from_numpy(np.array([gradclipvalue], dtype=np.float)).float()
        ctx.save_for_backward(x, h, weight_hh, h0)
        return h
    
    @staticmethod
    def backward(ctx, grad_h):
        gradclipvalue = 0
        # x, h, weight_hh, h0 = self.saved_tensors
        x, h, weight_hh, h0, = ctx.saved_tensors

        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        hidden_size = x.size(-1)#self.hidden_size
        ncols = batch*hidden_size
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        grad_x = x.new(*x.size())
        grad_weight_hh = x.new(hidden_size).zero_()
        grad_h0 = x.new(batch, hidden_size)  
        stream, _, bwd_func = GetFunctions().get_functions()
        # stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func
        FUNC(args=[
            x.contiguous().data_ptr(),
            weight_hh.contiguous().data_ptr(),
            h0.contiguous().data_ptr(),
            h.contiguous().data_ptr(),
            grad_h.contiguous().data_ptr(),
            length,
            batch,
            hidden_size,
            grad_x.contiguous().data_ptr(),
            grad_weight_hh.contiguous().data_ptr(),
            grad_h0.contiguous().data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        if gradclipvalue>0:
            grad_x.clamp_(-gradclipvalue,gradclipvalue)
            grad_weight_hh.clamp_(-gradclipvalue,gradclipvalue)
            grad_h0.clamp_(-gradclipvalue,gradclipvalue)
        return grad_x, grad_weight_hh, grad_h0



class IndRNN_onlyrecurrent(nn.Module):
    def __init__(self, hidden_size,gradclipvalue=0,
                 hidden_max_abs=None, recurrent_init=None):
        super(IndRNN_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.weight_hh = Parameter(torch.Tensor(hidden_size))   
        self.gradclipvalue=gradclipvalue         
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.uniform_(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, inputs, h0=None):
        
        if type(inputs) is dict:
            input = inputs[0]
            h0 = inputs[1]
        else:
            input = inputs

        assert input.dim() == 2 or input.dim() == 3        
        if h0 is None:
            h0 = input.data.new(input.size(-2),input.size(-1)).zero_()
        
        elif (h0.size(-1)!=input.size(-1)) or (h0.size(-2)!=input.size(-2)):
            raise RuntimeError(
                'The initial hidden size must be equal to input_size. Expected {}, got {}'.format(
                    h0.size(), input.size()))
      
        IndRNN_Compute = IndRNN_Compute_GPU().apply
     
        return IndRNN_Compute(input, self.weight_hh, h0)
