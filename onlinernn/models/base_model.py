from abc import ABC
import numpy as np
import torch

# ----------------------------------------------
"""
    Abstract class defining the APIs for model classes
"""
# ----------------------------------------------
class BaseModel(ABC):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        np.random.seed(42)
        self.opt = opt
        self.istrain = opt.istrain
        self.lr = opt.lr
        self.seq_len = opt.seq_len
        self.num_layers = opt.num_layers
        self.init_mode = opt.init_mode
        self.hidden_size = opt.hidden_size
        self.input_size = opt.feature_shape
        self.output_size = opt.n_class
        self.T = opt.T_
        self.device = opt.device
        self.permute_row = opt.permute_row
        self.permute_idx = torch.Tensor(np.random.permutation(self.seq_len).astype(np.float64)).long()
    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
        return self.class_name()

    # ----------------------------------------------
