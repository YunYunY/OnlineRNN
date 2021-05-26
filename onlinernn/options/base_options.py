import argparse
import torch

class BaseOptions:
    """This class defines options used during both training and test time."""

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # torch.manual_seed(42)

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
        parser.add_argument('--eval_freq', type=int, default=1, help='validation frequency while training')

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
            help="shuffle data for dataloader",
        )
        parser.add_argument(
            "--mnist_standardize",
            type=str,
            default="zeromean",
            help="the way to standardize MNIST data",
        ) #[originalmean, zeromean]

        parser.add_argument(
            "--subsequene",
            action="store_true",
            default=False,
            help="slice data to get new batches, if orginal data is 100, slice will get 5 * 20 if 5 is slice interval",
        )
        parser.add_argument(
            "--subseq_size", type=int, default=8, help="length of subsequence"
        ) 

        parser.add_argument(
            "--add_noise", action="store_true", default=False, help="add noise to data"
        ) 
        # -------------------------------------------------------
        # model parameters
        # -------------------------------------------------------

        parser.add_argument(
            "--predic_task", type=str, default="Binary", help="prediction task, decide final layer in RNN"
        ) # ["Binary", "Softmax"]

        parser.add_argument(
            "--single_output", action="store_true", default=True, help="prediction single value RNN"
        ) 
        parser.add_argument(
                    "--num_layers", type=int, default=1, help="number of layers in RNN"
                )
        parser.add_argument(
                    "--hidden_size", type=int, default=80 , help="number of neurons in hidden state"
                ) # 256
        parser.add_argument(
            "--init_mode", type=str, default="Zeros", help="method to initialize first hidden state"
        ) #["Zeros",  "Random"]

        parser.add_argument(
            "--LSTM",
            action="store_true",
            default=False,
            help="use LSTM cell or not",
        )

        parser.add_argument('--clip_grad', action='store_true', default=False)

        # FGSM
        parser.add_argument('--iterT', type=int, default=1, help='iterT for FGSM optimizer')
        parser.add_argument('--iterB', type=int, default=0, help='inside sample batches for FGSM') # run exp 10

        # TBPTT
        parser.add_argument('--for_trunc', type=list, default=30, help='truncate parameter forward steps for TBPTT ') 
        parser.add_argument('--back_trunc', type=list, default=20, help='truncate parameter backward steps for TBPTT ') 

        # indRNN sequential MNIST
        parser.add_argument('--MAG', type=float, default=5.0)
        parser.add_argument('--u_lastlayer_ini', action='store_true', default=True)
        parser.add_argument('--constrain_U', action='store_true', default=True)
        parser.add_argument('--constrain_grad', action='store_true', default=True)
        parser.add_argument('--model', type=str, default='plainIndRNN')
        parser.add_argument('--bn_location', type=str, default='bn_after')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pThre', type=int, default=100)
        parser.add_argument('--U_bound', type=float, default=0.0)
        parser.add_argument('--decayfactor', type=float, default=1e-4,help='lr')
        parser.add_argument('--gradclipvalue', type=float, default=10, help='gradclipvalue')
       

        # -------------------------------------------------------
        # additional parameters 
        # -------------------------------------------------------

        parser.add_argument(
            "--test_batch",
            action="store_true",
            help="if specified, run test after every batch",
        )

        parser.add_argument(
            "--verbose",
            action="store_false",
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
