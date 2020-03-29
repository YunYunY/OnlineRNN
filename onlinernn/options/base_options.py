from options.base_options import BaseOptions


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
            "--niter", type=int, default=19, help="# of iter at starting learning rate"
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

        # CycleGAN related parameters
        parser.add_argument("--lambda_A", type=float, default=100, help="lambda_A")
        parser.add_argument("--lambda_B", type=float, default=100, help="lambda_B")
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # Add two parameters infront of GAN loss for CycleGAN
        parser.add_argument("--betaG_A", type=float, default=1, help="betaG_A")
        parser.add_argument("--betaG_B", type=float, default=1, help="betaG_B")
        parser.add_argument("--betaD_A", type=float, default=1, help="betaD_A")
        parser.add_argument("--betaD_B", type=float, default=1, help="betaD_B")
        # DCGAN
        parser.add_argument("--dataname", default="MNISTresize")
        parser.add_argument(
            "--model_size", type=int, default=64, help="model size for DCGAN"
        )

        return parser
