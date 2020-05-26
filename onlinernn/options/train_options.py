from onlinernn.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--continue_train",
            action="store_true",
            default=False,
            help="continue training from a saved step",
        ) 

        parser.add_argument(
            "--batch_size", type=int, default=128, help="input batch size"
        ) # 64 
        # model training log step
        parser.add_argument(
            "--log",
            action="store_false",
            default=True,
            help="output the training log or not",
        )
        parser.add_argument(
            "--log_freq",
            type=int,
            default=1,
            help="frequency of print the log while training",
        )

        parser.add_argument(
            "--epoch_count",
            type=int,
            default=0,
            help="the starting epoch count, we save the model by every save_epoch_freq",
        )

        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=50,
            help="frequency of saving checkpoints at the end of epochs",
        )
     
        parser.add_argument(
            "--epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        parser.add_argument(
            "--load_iter",
            type=int,
            default="0",
            help="change this and change niter. Which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]",
        )

        # training parameters
        # niter and niter_decay only works when model requires lr decay, otherwise they are only used to calculate total number of epoch
        # both of indexes start from 0
        parser.add_argument("--niter", type=int, default=199, help="# of iter at starting learning rate decay"
        )
        parser.add_argument(
            "--niter_decay",
            type=int,
            default=0,
            help="# of iter to linearly decay learning rate to zero or period for step lr",
        ) 
        parser.add_argument('--end_rate', type=float, default=1e-6) # end training once accuracy lower than this 

        parser.add_argument(
            "--lr_policy", type=str, default="step", help="the learning rate policy"
        )  # 'linear', 'step'
        parser.add_argument('--lrgamma', type=float, default=0.1) # lr decay rate

        parser.add_argument(
            "--endless_train",
            action="store_true",
            default=False,
            help="reduce the lr by a factor after certain condition, until converge. Otherwise keep training",
        ) 
     
        parser.add_argument(
            "--lr", type=float, default=0.001, help="initial learning rate for adam"
        ) # 0.001
        parser.add_argument('--optimizer', type=str, default='FGSM_Adam', help='Optimizer option'
        ) #["SGD, Adam, SGD_Momentum, FGSM_Adam, FGSM_RMSProp"]

        parser.add_argument('--reg_lambda', type=float, default=0, help='Lambda for StopBP')

        return parser
