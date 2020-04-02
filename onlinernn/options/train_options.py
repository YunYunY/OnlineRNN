from onlinernn.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
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
            default=10,
            help="frequency of print the log while training",
        )
        # network saving and loading parameters
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
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
     
        # training parameters
        # niter and niter_decay only works when model requires lr decay
        # otherwise they are only used to calculate total number of epoch
        # both of indexes start from 0
        parser.add_argument(
            "--niter", type=int, default=1, help="# of iter at starting learning rate"
        )
        parser.add_argument(
            "--niter_decay",
            type=int,
            default=0,
            help="# of iter to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--lr_policy", type=str, default="linear", help="the learning rate policy"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument('--state_update', action="store_true", default=False, help='FPP: w/ or w/o state update')
        return parser
