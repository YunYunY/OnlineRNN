import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from onlinernn.models.networks import SimpleRNN, StepRNN
from onlinernn.models.base_model import BaseModel
from onlinernn.models.fgsm import FGSM

class VanillaRNN(BaseModel):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(VanillaRNN, self).__init__(opt)
        # self.loss_method = "vanilla"
        # self.optimizers = ([])  # define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.
        self.model_names = ["rnn_model"]
        self.model_method = "Vanilla" # the way to construct model
        self.loss_method = "MSE" # ["BCELogit" "CrossEntropy" "MSE"]

  
    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        if self.model_method == "Vanilla":
            self.rnn_model = SimpleRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        elif self.model_method == "StepRNN":
            self.rnn_model = StepRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)

        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()
        if (self.device.type == "cuda") and (self.opt.ngpu > 1):
            print('Run parallel')
            self.rnn_model = nn.DataParallel(self.rnn_model, list(range(self.opt.ngpu)))


    def init_loss(self):
        """
        Define Loss functions
        """
        if self.loss_method == "MSE":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_method == "CrossEntropy":
            self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)
        elif self.loss_method == "BCELogit":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        

    def init_optimizer(self):
        """
        Setup optimizers
        """
        # self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        if self.opt.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=self.lr, weight_decay=0.0001)

        elif self.opt.optimizer == 'FGSM':
            self.optimizer = FGSM(self.rnn_model.parameters(), lr=self.lr, iterT=self.T)
    # ----------------------------------------------

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def setup(self):
        """
            Setup the schedulers or load the model
        """
        # TODO LR SCHEDULER MAY NEED TO BE STORED FOR RESUME
#https://discuss.pytorch.org/t/why-doesnt-resuming-work-properly-in-pytorch/19430/4
        # if self.istrain:
            # self.schedulers = [
            #     get_scheduler(optimizer, self.opt) for optimizer in self.optimizers
            # ]
        if (not self.istrain) or self.opt.continue_train:
            load_prefix = self.opt.load_iter if self.opt.load_iter > 0 else self.opt.epoch
            print(f'Load the {load_prefix} epoch network')
            self.load_networks(load_prefix)
            self.test_acc = 0

        self.print_networks(self.opt.verbose)
    
    # ----------------------------------------------
    def load_networks(self, load_prefix):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{load_prefix}_{name}_T{self.T}.pth"
                load_path = os.path.join(self.result_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

   
            load_filename = f"{load_prefix}_optimizer_T{self.T}.pth"
            load_path = os.path.join(self.result_dir, load_filename)
            optimizer = getattr(self, "optimizer")
    
            print("loading the optimizer from %s" % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)
    # ----------------------------------------------
    def set_test_input(self):
        self.set_input()

    def set_input(self):
        self.inputs, self.labels = self.data
        self.inputs = self.inputs.view(-1, self.seq_len, self.input_size).to(self.device)
        self.labels = self.labels.view(-1).to(self.device)
        # update batch 
        self.batch_size = self.labels.shape[0]



    def set_output(self):
        self.losses = 0
        self.reg1 = 0
        self.reg2 = 0
        self.train_acc = 0

    # -------------------------------------------------------
    # Initialize first hidden state      

    def init_states(self):
        # self.states = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        self.states = torch.zeros((self.batch_size, self.num_layers, self.hidden_size)).to(self.device)

    def track_grad_flow(self, named_parameters):
        # Reference: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
        # ave_grads = []
        # layers = []
        for n, p in named_parameters:
            if "weight_hh_l0" in n:
                if p.requires_grad:
                    # layers.append(n)
                    # ave_grads.append(p.grad.abs().mean())
                    self.weight_hh = p.grad.abs().mean()

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer.zero_grad()
        self.init_states()
        
        if self.model_method == "Vanilla":
            outputs, _ = self.rnn_model(self.inputs, self.states)
            outputs = outputs.to(self.device)
        elif self.model_method == "StepRNN":
            outputs, states, state_start, state_final  = self.rnn_model(self.inputs, self.states)
            outputs, state_start, state_final = outputs.to(self.device), state_start.to(self.device), state_final.to(self.device)
            # state_start = states[1][1]
            # state_final = states[-1][1]
            # outputs, state_final, state_start = outputs.to(self.device), state_final.to(self.device), state_start.to(self.device)
        
            # calculate the magnitude of gradient of the last layer hT wrt h1
            # self.state_grad = torch.autograd.grad(state_final.sum(), state_start, retain_graph=True)
            # self.state_grad = torch.norm(self.state_grad)

   
        # print(outputs.shape)
        # print(self.labels.shape)
        self.labels = self.labels.float()
        self.loss = self.criterion(outputs, self.labels)
        self.loss.backward()

        self.track_grad_flow(self.rnn_model.named_parameters())

        self.optimizer.step()

        self.losses += self.loss.detach().item()
        self.train_acc += self.get_accuracy(outputs, self.labels, self.batch_size)



    def training_log(self, batch):
        """
        Save gradient
        """
        np.savez(
            self.result_dir + "/" + str(batch) + "_weight_hh.npz", weight_hh=self.weight_hh.cpu())
        

   # ----------------------------------------------

    def get_accuracy(self, logit, target, batch_size):
        """ 
        Obtain accuracy for training round 
        """
        # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        # accuracy = 100.0 * corrects/batch_size
        pred = logit >= 0.5
        truth = target >= 0.5
        accuracy = 100.* pred.eq(truth).sum()/batch_size 

        return accuracy.item()

    def save_losses(self, epoch):
        # if self.opt.verbose:
        #     print( "Loss of epoch %d / %d "
        #         % (epoch, self.loss))
        np.savez(
            self.loss_dir + "/" + str(epoch) + "_losses.npz",
            losses=self.losses     )
    # ----------------------------------------------
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)

        """

        # Save networks
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_%s_T%s.pth" % (epoch, name, self.T)
                save_path = os.path.join(self.result_dir, save_filename)
                net = getattr(self, name)
                if (self.opt.device.type == "cuda") and (self.opt.ngpu > 0):
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda()
                else:
                    torch.save(net.cpu().state_dict(), save_path)

        # Save optimizers

        save_filename = "%s_optimizer_T%s.pth" % (epoch, self.T)
        save_path = os.path.join(self.result_dir, save_filename)
        optimizer = getattr(self, "optimizer")
        torch.save(optimizer.state_dict(), save_path)


    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states)
            outputs = outputs.to(self.device).detach()
            self.test_acc += self.get_accuracy(outputs, self.labels, self.batch_size)
      
            # self.save_result()

    # ----------------------------------------------
    def get_test_acc(self, n):
        print(f"Calculate accuracy after {n} loads")
        self.test_acc = self.test_acc / (n)
        print(f"Test accuracy is {self.test_acc}")
        self.save_result()
    # ----------------------------------------------

    def save_result(self):
        # Plot loss
        load_prefix = self.opt.load_iter if self.opt.load_iter > 0 else self.opt.epoch
        np.savez(self.result_dir + "/test_epoch_" + str(load_prefix) + "_T"+ str(self.T)+ ".npz", accuracy=self.test_acc)
        # imshow(
        #     torch.reshape(
        #         self.test_fake, (self.plt_n * self.plt_n, 1, self.x_dim, self.x_dim)
        #     ),
        #     self.result_dir,
        #     self.plt_n,
        #     "Laststep_G.png",
        # )