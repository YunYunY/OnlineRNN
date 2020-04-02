from abc import ABC


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
        self.state_update = opt.state_update
        self.lr = opt.lr
        self.hidden_size = opt.hidden_size
        self.input_size = opt.feature_shape
        self.output_size = opt.n_class
        self.device = opt.device
        self.T = opt.T
        self.batch_size = opt.batch_size  # B

    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
        return self.class_name()

    # ----------------------------------------------
