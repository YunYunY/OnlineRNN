from abc import abstractmethod
import torch
import torch.nn as nn
import os
import time
import numpy as np


class Setting:
    """
        setting is an abstract class for experiment settings.
        It defines the basic APIs for experiment setting classes.
    """

    def __init__(self, opt):
        self.opt = opt
        self.istrain = opt.istrain
        # self.continue_train = opt.continue_train
        # Decide which device we want to run on
        opt.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
        )
        opt.n_epochs = (
            self.opt.niter + self.opt.niter_decay
        )  # If the total is 20, n_epochs is 19. Index start from 0
        # Output related
        self.log = opt.log
        self.log_freq = opt.log_freq

    # ----------------------------------------------

    def create_final_dir(self):
        """
            Check if the results folder exists
            If not, create the directory
        """
        self.result_dir = os.path.join("result", self.model.name, self.dataset.name)
        os.makedirs(self.result_dir, exist_ok=True)
        self.loss_dir = os.path.join(self.result_dir, "loss")
        os.makedirs(self.loss_dir, exist_ok=True)

    # ----------------------------------------------
    def setup(
        self,
        dataset,  # the dataset object of the experiment
        model,  # the deep learning model object in the experiment
    ):
        """
            setup everything before running experiment
            assign the proper properties to model and dataset
        """

        self.model = model
        # setup dataset
        self.dataset = dataset
        # setup model
        self.model.dataname = self.dataset.name
        # output related
        self.model.plt_n = 8
        self.create_final_dir()
        self.model.result_dir = self.result_dir
        self.model.loss_dir = self.loss_dir

    # ----------------------------------------------
    @abstractmethod
    def run(self):
        """
            The API for experiment run
        """
        raise NotImplementedError


# ----------------------------------------------


class RNN(Setting):
    """
    Define model structure and running process
    """

    def __init__(self, opt):
        super(RNN, self).__init__(opt)

    def run(self):

        # Build network structure & Initialize hidden layer
        self.model.init_net()
        # Setup loss
        self.model.init_loss()
        # Setup optimizer
        self.model.init_optimizer()
        # Load networks; create schedulers
        self.model.setup()
        print("Starting Training Loop...")
        self.model.datasize = len(self.dataset.dataloader)
        total_iters = 0  # the total number of training iterations

        # for epoch in range(self.n_epochs):
        for epoch in range(
            self.opt.epoch_count, self.opt.n_epochs + 1):
            # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

            # Setup output related
            self.model.set_output()

            # For each batch
            epoch_start_time = time.time()  # timer for entire epoch

            for i, data in enumerate(self.dataset.dataloader):

                total_iters += self.opt.batch_size
                # Setup input
                self.model.data = data  # load training data
                self.model.set_input()
                # Forward, backward, update network weights
                self.model.train() 
                # Output training stats
                if self.log and i % self.log_freq == 0:
                    self.model.training_log(i, epoch)

            # if (
            #     epoch + 1
            # ) % self.opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            #     print(
            #         "saving the model at the end of epoch %d, iters %d"
            #         % (epoch, total_iters)
            #     )
            #     self.model.save_networks(epoch)

            # if epoch == self.opt.n_epochs:
                # self.model.save_networks("latest")
            print(
                "End of epoch %d / %d \t Time Taken: %d sec"
                % (epoch, self.opt.n_epochs, time.time() - epoch_start_time)
            )
            # self.model.update_learning_rate()  # update learning rates at the end of every epoch.
            # Save losses at each epoch
            self.model.save_losses(epoch)
        # Plot loss at the end of the run
        # self.model.visualize()


    def test(self):
        """
        Test a model
        """
        # Build network structure
        self.model.init_net()
        self.model.init_optimizer()
        # Load the model and losses
        self.model.setup()

        """
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        """
        if self.opt.eval:
            self.model.eval()
        for i, data in enumerate(self.dataset.dataloader):
            if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
                break
            # Setup input
            self.model.data = data  # load training data
            if hasattr(self.dataset, "flipdataloader"):
                self.model.flipdata = next(iter(self.dataset.flipdataloader))
            self.model.set_test_input()  # unpack data from data loader
            self.model.test()  # run inference
