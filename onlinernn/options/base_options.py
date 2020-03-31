import argparse
import torch


class BaseOptions:
    """This class defines options used during both training and test time."""

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # torch.manual_seed(999)

        """Define the common options that are used in both training and test."""
        # basic parameters

        # parser.add_argument("--istrain", action='store_false', default=True, help="train the model or test")
        parser.add_argument(
            "--istrain",
            action="store_true",
            default=False,
            help="train the model or test",
        )

        parser.add_argument("--num_test", type=int, default=1, help="# of test epoch")
        parser.add_argument(
            "--batch_size", type=int, default=64, help="input batch size"
        )
        parser.add_argument(
            "--num_threads", type=int, default=2, help="# threads for loading data"
        )
        parser.add_argument("--ngpu", type=int, default=1, help="# of GPUs")
        parser.add_argument("--test_case", default=None, help="run test case or not")
        # task
        parser.add_argument(
            "--taskid", type=int, default=0, help="the experiment task to run"
        )
        # data parameters

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
            help="shuffle data",
        )

        parser.add_argument(
            "--nc", type=int, default=1, help="# of channels for input data"
        )

        # model parameters
        # parser.add_argument(
        #     "--n_epochs", type=int, default=300, help="# of maximum epoch"
        # )

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument(
            "--eval", action="store_true", help="use eval mode during test time."
        )

        # additional parameters
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
            help="which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]",
        )
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
        # return parser.parse_args()
        return parser.parse_known_args()

    def parse(self):
        opt = self.gather_options()[0]
        self.opt = opt
        return self.opt
