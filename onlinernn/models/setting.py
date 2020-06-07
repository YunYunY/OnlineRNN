from abc import abstractmethod
import torch
import torch.nn as nn
import os
import time
import numpy as np
import shutil


class Setting:
    """
        setting is an abstract class for experiment settings.
        It defines the basic APIs for experiment setting classes.
    """

    def __init__(self, opt):
        self.opt = opt
        self.istrain = opt.istrain
        self.continue_train = opt.continue_train
        # Decide which device we want to run on
        # opt.device = torch.device(
        #     "cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
        # )
        opt.device = torch.device(
            "cuda" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu"
        )
        print("Number of available GPUs: %d" %torch.cuda.device_count())
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
        # self.result_dir = os.path.join("result", self.model.name, self.dataset.classname, "T"+str(self.model.T), self.opt.optimizer, str(self.opt.taskid))
        self.result_dir = os.path.join("result", self.model.name, self.dataset.name, "T"+str(self.model.T), self.opt.optimizer, str(self.opt.taskid))

        os.makedirs(self.result_dir, exist_ok=True)
        # if os.path.exists(self.result_dir):
        #     shutil.rmtree(self.result_dir)
        # os.makedirs(self.result_dir)
        print(f"Output folder {self.result_dir}")
        self.loss_dir = os.path.join(self.result_dir, "loss_acc")
        os.makedirs(self.loss_dir, exist_ok=True)

    # ----------------------------------------------
    def setup(
        self,
        dataset,  # the dataset object of the experiment
        model,  # the deep learning model object in the experiment
        dataset_test, # test dataset
    ):
        """
            setup everything before running experiment
            assign the proper properties to model and dataset
        """

        self.model = model
        # setup dataset
        self.dataset = dataset
       
        self.dataset_test = dataset_test
        # setup model
        self.model.dataname = self.dataset.name
        self.model.trainsize = len(self.dataset)
        self.model.trainloadersize = len(self.dataset.dataloader)
        self.model.dataset = self.dataset
     
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


    def train_svrg(self):
        self.model.global_train()


    def train_sgd(self):
        for i, data in enumerate(self.dataset.dataloader):
            self.total_iters += self.opt.batch_size
            self.model.total_batches += 1
            self.model.data = data 
            self.model.set_input()

            # ---------------------------------------------------

            # Forward, backward, update network weights
            self.model.train() 
            # Save gradients
            if self.log and (self.model.total_batches-1)%self.model.T == (self.model.T-1):
                self.model.training_log(self.model.total_batches)
            if self.opt.test_batch:
                print("Epoch %d | End of batch %d | Time Taken: %d sec | Loss: %.4f"
                % (self.epoch, self.model.total_batches, time.time() - self.epoch_start_time, self.model.loss))
                # self.model.set_test_output()
                # for i, data in enumerate(self.dataset_test.dataloader):
                #     self.model.data = data 
                #     self.model.set_test_input()  # unpack data from data loader
                #     self.model.test()  # run inference
                # self.model.get_test_acc() # calculate and save global acc
                
                # self.model.save_test_acc(self.model.total_batches)


    def run(self):
        global_start_time = time.time()  # timer for entire epoch

        # Build network structure
        self.model.init_net()

        # Setup loss
        self.model.init_loss()
        # Setup optimizer
        self.model.init_optimizer()
        # Load networks; create schedulers
        self.model.setup()
  
        print("Start Training Loop...")
        self.total_iters = 0  # the total number of training iterations
        self.model.total_batches = 0 # the total number of batchs
        self.model.max_test_acc = 0
        self.model.min_test_mse = float('inf')
        self.model.trace_fgsm = []

#---------------------------------------------
        for epoch in range(
            self.opt.epoch_count, self.opt.n_epochs + 1):

            self.epoch = epoch
            self.epoch_start_time = time.time()  # timer for entire epoch
            self.model.set_output()

            if self.opt.optimizer == "SVRG":
                self.train_svrg()
            else:
                self.train_sgd()
         
            if (epoch + 1) % self.opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(
                    "saving the model at the end of epoch %d, iters %d"
                    % (epoch, self.total_iters))
                self.model.save_networks(epoch)

            if epoch == self.opt.n_epochs:
                self.model.save_networks("latest")
            # ----------------------------------------------
            self.model.save_losses(epoch)
            if self.opt.verbose:
                print("End of epoch %d / %d | Time Taken: %d sec | Loss: %.4f | Train Accuracy: %.2f"
                    % (epoch, self.opt.n_epochs, time.time() - self.epoch_start_time, self.model.losses, self.model.train_acc))
            # ----------------------------------------------
            if not self.opt.test_batch and self.opt.eval_freq != 0: # evaluate 
                self.model.set_test_output()
                for i, data in enumerate(self.dataset_test.dataloader):
                    self.model.data = data 
                    self.model.set_test_input()  # unpack data from data loader
                    self.model.test()  # run inference
                self.model.get_test_acc() # calculate and save global acc
                self.model.save_test_acc(epoch)
            print(f'Total training time is {time.time() - global_start_time}')
            lr = self.model.update_learning_rate()  # update learning rates at the end of every epoch.
            print('learning rate = %.10f' % lr)
            if self.opt.endless_train and lr < self.opt.end_rate:
                break

        print(f'Total batch is { self.model.total_batches}')
        print(f"Output folder {self.result_dir}")
        # self.save_temp()
        

    def save_temp(self):
        print(self.model.trace_fgsm)
        np.savez(self.loss_dir + "temp_dis.npz", dis = self.model.trace_fgsm)

    def test(self):
        """
        Test a model
        """

        # Build network structure
        self.model.init_net()
        self.model.init_optimizer()
        # Load the model and losses
        self.model.setup()
        self.model.set_test_output()
        
        for i, data in enumerate(self.dataset.dataloader):
            # if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
            #     break
            # Setup input
            self.model.data = data 
            self.model.set_test_input()  # unpack data from data loader
            self.model.test()  # run inference
        self.model.get_test_acc() # calculate and save global acc
