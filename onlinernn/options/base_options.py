import argparse
import torch

class BaseOptions:
    """This class defines options used during both training and test time."""

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        torch.manual_seed(42)

        # -------------------------------------------------------
        # basic parameters
        # -------------------------------------------------------

        parser.add_argument(
            "--istrain",
            action="store_true",
            default=False,
            help="train the model or test",
        )
        parser.add_argument("--num_test", type=int, default=1, help="# of test epoch")

        parser.add_argument(
            "--num_threads", type=int, default=2, help="# threads for loading data"
        )
        parser.add_argument("--ngpu", type=int, default=1, help="# of GPUs")
        parser.add_argument("--test_case", default=None, help="run test case or not")
        parser.add_argument(
            "--taskid", type=int, default=0, help="the experiment task to run"
        )
        # -------------------------------------------------------
        # data parameters
        # -------------------------------------------------------

        parser.add_argument(
            "--download_data",
            action="store_true",
            default=False,
            help="download data",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            default=True,
            help="shuffle MNIST data",
        )
        parser.add_argument(
            "--mnist_standardize",
            type=str,
            default="zeromean",
            help="the way to standardize MNIST data",
        ) #[originalmean, zeromean]

    
        # -------------------------------------------------------
        # model parameters
        # -------------------------------------------------------

        # Dropout and Batchnorm have different behavioir during training and test.
        parser.add_argument(
                    "--num_layers", type=int, default=1, help="number of layers in RNN"
                )
        parser.add_argument(
                    "--hidden_size", type=int, default=80 , help="number of neurons in hidden state"
                ) # 256
        parser.add_argument(
            "--init_mode", type=str, default="Zeros", help="Method to initialize first hidden state"
        ) #["Zeros",  "Random"]
        parser.add_argument('--T', type=list, default=[1], help='Truncate parameter for TBPTT or iterT for VanillaRNN') #[7, 14, 21, 28] for truncation [1, 3, 5] for fgsm
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer option') #["Adam, FGSM"]



        # -------------------------------------------------------
        # additional parameters 
        # -------------------------------------------------------

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if specified, print more debugging information",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        return parser.parse_known_args()

    def parse(self):
        opt = self.gather_options()[0]
        self.opt = opt
        return self.opt
