import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functools 
import random
import copy
from onlinernn.models.networks import SimpleRNN, ODE_Vanilla, get_scheduler
from onlinernn.models.base_model import BaseModel
from onlinernn.models.indrnn_utils import clip_gradient, clip_weight
from onlinernn.tests.test_utils import show_shift

class VanillaRNN(BaseModel):
    def __init__(self, opt):
    # def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):
        super(VanillaRNN, self).__init__(opt)
        # self.loss_method = "vanilla"
        self.model_names = ["rnn_model"]
        self.model_method = "Vanilla" # the way to construct model
        if opt.predic_task in ['Binary', 'Logits']:
            self.loss_method = "MSE" # ["BCELogit" "CrossEntropy" "MSE"]
            self.init_state_C = np.sqrt(3 / (2 * opt.hidden_size))
          
        else:
            self.loss_method = "CrossEntropy" # ["BCELogit" "CrossEntropy" "MSE"]
  
    def init_net(self):
        """
        Initialize model
        Dropout works as a regularization for preventing overfitting during training.
        It randomly zeros the elements of inputs in Dropout layer on forward call.
        It should be disabled during testing since you may want to use full model (no element is masked)
        
        """
        if self.model_method == "Vanilla":
            # self.rnn_model = SimpleRNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        #     self.rnn_model = SimpleRNN(self.opt).to(self.device)
            self.rnn_model = ODE_Vanilla(self.opt).to(self.device)

        if self.opt.optimizer == 'SVRG':
            self.model_snapshot = copy.deepcopy(self.rnn_model)

        # explicitly state the intent
        if self.istrain:
            self.rnn_model.train()
        else:
            self.rnn_model.eval()
        if (self.device.type == "cuda") and (self.opt.ngpu > 1):
            print('Run parallel')
            self.rnn_model = nn.DataParallel(self.rnn_model, list(range(self.opt.ngpu)))


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
        if self.istrain and self.opt.niter_decay != 0:
            # self.schedulers = [get_scheduler(self.optimizer, self.opt)]
            self.schedulers = [
                get_scheduler(optimizer, self.opt) for optimizer in self.optimizers
            ]
         
        if (not self.istrain) or self.opt.continue_train:
            load_prefix = self.opt.load_iter if self.opt.load_iter > 0 else self.opt.epoch
            print(f'Load the {load_prefix} epoch network')
            self.load_networks(load_prefix)

        self.print_networks(self.opt.verbose)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if self.istrain and self.opt.niter_decay != 0:
            for scheduler in self.schedulers:
                if self.opt.lr_policy == 'plateau':
                    scheduler.step(self.metric)
                else:
                    scheduler.step()

            # lr = self.optimizer.param_groups[0]['lr']
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr


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
              

        for i, optimizer in enumerate(self.optimizers):
            load_filename = "%s_optimizer_T%s_optimizer_%s.pth" % (load_prefix, self.T, i)
            load_path = os.path.join(self.result_dir, load_filename)
            # optimizer = getattr(self, "optimizer")
            print("loading the optimizer from %s" % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)
        # if self.opt.niter_decay != 0:
        #     for i, scheduler in enumerate(self.schedulers):
        #         load_filename = "%s_scheduler_T%s_scheduler_%s.pth" % (load_prefix, self.T, i)
        #         load_path = os.path.join(self.result_dir, load_filename)
        #         print("loading the scheduler from %s" % load_path)
        #         state_dict = torch.load(load_path, map_location=self.device)
            
        #         if hasattr(state_dict, "_metadata"):
        #             del state_dict._metadata
        #         scheduler.load_state_dict(state_dict)
    # ----------------------------------------------
    def set_test_input(self):
        self.set_input()

    def set_input(self):
        self.inputs, self.labels = self.data
      
        if self.add_noise:
            
            noise = torch.randn((self.inputs.shape[0], self.seq_len, self.inputs.shape[2])) * 1. + 0.
            noise[:, 0: self.inputs.shape[1], :] =  self.inputs
            self.inputs = noise
        # data = noise
        # data = data[:, 0:40, :]
        # data = data.reshape(-1, 40, 3, 32)
        # data = data.permute(0, 2, 1, 3)
    
        # result_dir = "result/dataset_test/CIFARNoise"
        # os.makedirs(result_dir, exist_ok=True)
        # # visually check image after shifting
        # show_shift(data, 8, result_dir, "CIFARNoise.png")
        # exit(0)
        self.inputs = self.inputs.view(-1, self.seq_len, self.input_size).to(self.device)
        self.batch_size = self.labels.shape[0]  # update batch 
        self.labels = self.labels.view(-1).to(self.device)
        if self.opt.predic_task in ['Binary', 'Logits']:
            self.labels = self.labels.float()

    def set_output(self):
        # initialize values at the beginning of each epoch
        self.losses = []
        self.reg1 = 0
        self.reg2 = 0
        self.train_acc = []

    def set_test_output(self):
        self.test_acc = []
    # -------------------------------------------------------
    # Initialize first hidden state      

    def init_states(self):
      

        # if self.opt.predic_task in ['Logits']:
        #     self.states = torch.FloatTensor(self.batch_size, self.num_layers, self.seq_len, self.hidden_size).uniform_(-self.init_state_C, self.init_state_C).to(self.device)
        # else:
        self.states = torch.zeros((self.batch_size, self.hidden_size)).to(self.device)
        
        # self.states = torch.zeros((self.T+1, self.batch_size, self.hidden_size)).to(self.device)
        # self.states = torch.zeros((self.batch_size, self.seq_len, self.hidden_size)).to(self.device)
        # self.states = torch.FloatTensor(self.batch_size, self.seq_len, self.hidden_size).uniform_(-self.init_state_C, self.init_state_C).to(self.device)

    def track_grad_flow(self, named_parameters):
        # Reference: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
  
        for n, p in named_parameters:
            if "weight_hh_l0" in n:
                if p.requires_grad:

                    self.trace_fgsm.append(p.dis.detach().item())
                    # self.weight_hh = p.buf.abs().mean()
                    # self.weight_hh = p.grad.abs().mean()
                    # self.weight_hh = torch.norm(p.grad)
               
    # -------------------------------------------------------
    # Train model 
    def inner_sample_train(self):
        sample_seed = random.sample(range(0, self.trainsize-self.opt.batch_size), self.T)  

        iter = 0
        # load subset of data 
        for i in sample_seed:
            iter += 1
            subset_loader = self.dataset.subset_loader(i)
            for _, data in enumerate(subset_loader):
                self.data = data
                self.set_input()
            self.forward()
            self.backward()
            self.optimizer.step(iter, sign_option=True)

        # After last iterT, track Delta w, loss and acc
        self.track_grad_flow(self.rnn_model.named_parameters())
        self.losses.append(self.loss.detach().item())
        self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))

    def forward(self):
        """
        Forward path 
        """
        self.optimizer.zero_grad()
        self.init_states()
        if self.model_method == "Vanilla":
            outputs, _ = self.rnn_model(self.inputs, self.states)
            self.outputs = outputs.to(self.device)

    def backward(self):
        """
        Backward path 
        """
        
        self.loss = self.criterion(self.outputs, self.labels)   

        self.loss.backward()
        

    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.backward()

     
        # first_iter = (self.total_batches-1)%self.T == 0 # if this is the first iter of inner loop
        # last_iter = (self.total_batches-1)%self.T == (self.T-1) # if this is the last step of inner loop

     
        '''
        if self.opt.global_u:
            #Backward
            for param1, param2, param3 in zip(self.rnn_model.parameters(), self.previous_net_sgd.parameters(), \
            self.previous_net_grad.parameters()): 
                if param1.grad is None:
                    continue 
                param1.data -= (self.lr) * (param1.grad.data - param2.grad.data + (1./self.batch_size) * param3.grad.data)
        '''
        
        if 'FGSM' in self.opt.optimizer:
            # if self.opt.iterB > 0:
            #     self.optimizer.step(self.total_batches, sign_option=True)
            # else:
            self.optimizer.step(self.total_batches)
        else:
           
            if self.opt.clip_grad:
                clip_gradient(self.rnn_model, self.gradientclip_value) 
        
            clip_weight(self.rnn_model, self.meta, self.device)

            self.optimizer.step()

        # if last_iter:
            # After last iterT, track Delta w, loss and acc
            # self.track_grad_flow(self.rnn_model.named_parameters())
        self.losses.append(self.loss.detach().item())
        self.train_acc.append(self.get_accuracy(self.outputs, self.labels, self.batch_size))


    def training_log(self, batch):
        """
        Save gradient and loss of the current single batch, not mean of the whole epoch 
        """   
        try:
            np.savez(self.loss_dir + "/batch_" + str(batch) + "_losses.npz", loss = self.loss.detach().item())
        except:
            np.savez(self.loss_dir + "/batch_" + str(batch) + "_losses.npz", loss = self.loss)

        # not every model has Delta w stored
        try:
            np.savez(self.result_dir + "/" + str(batch) + "_weight_hh.npz", weight_hh=self.weight_hh.cpu())
        except:
            pass
     


   # ----------------------------------------------

    def get_accuracy(self, logit, target, batch_size):
        """ 
        Obtain accuracy 
        """
        # y_pred_tag = torch.round(torch.sigmoid(logit))
        # corrects = (y_pred_tag.data == target.data).sum().float()
        # accuracy = 100.0 * corrects/batch_size
        if self.opt.predic_task == 'Binary':
            pred = logit >= 0.5
            truth = target >= 0.5
            accuracy = 100.* pred.eq(truth).sum()/batch_size
            
            
        elif self.opt.predic_task in['Logits', 'CM' ]:
            try:
                accuracy = self.loss.detach()
            except:
                accuracy = self.loss
                return accuracy
        else:
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch_size
        
        return accuracy.item()

    def save_losses(self, epoch):
        for name, param in self.rnn_model.named_parameters():
            if param.requires_grad:
                if 'alpha' in name or 'beta' in name or 'mu' in name or 'lr' in name:
                    print(name)
                    print(param.data)
        try:
            # calculate average loss for each batch and save
            self.losses = sum(self.losses) / len(self.losses)
            self.train_acc = sum(self.train_acc) / len(self.train_acc)
        except:
            # no losses calculated since iterT is bigger than the whole batch 
            self.losses = self.loss.detach().item()
            self.train_acc = self.get_accuracy(self.outputs, self.labels, self.batch_size)
        
        np.savez(
            self.loss_dir + "/epoch_" + str(epoch) + "_losses_train_acc.npz",
            losses = self.losses, train_acc = self.train_acc)
  
            

    def save_test_acc(self, epoch):
        if self.opt.predic_task == 'Logits':
            self.min_test_mse = min(self.min_test_mse, self.test_acc)
            print(f"Min test MSE is {self.min_test_mse}")
            np.savez(
            self.loss_dir + "/batch_" + str(epoch) + "_test_acc.npz",
            test_acc = self.test_acc)
        else:
            self.max_test_acc = max(self.max_test_acc, self.test_acc)
            print(f"Maximum test acc is {self.max_test_acc}")
            np.savez(
            self.loss_dir + "/epoch_" + str(epoch) + "_test_acc.npz",
            test_acc = self.test_acc)



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
        for i, optimizer in enumerate(self.optimizers):
            save_filename = "%s_optimizer_T%s_optimizer_%s.pth" % (epoch, self.T, i)
            save_path = os.path.join(self.result_dir, save_filename)
            # optimizer = getattr(self, "optimizer")
            torch.save(optimizer.state_dict(), save_path)

        
        # Save schedulers
        if self.opt.niter_decay != 0:
            for i, scheduler in enumerate(self.schedulers):
                save_filename = "%s_scheduler_T%s_scheduler_%s.pth" % (epoch, self.T, i)
                save_path = os.path.join(self.result_dir, save_filename)
                torch.save(scheduler.state_dict(), save_path)
      
 
    # ----------------------------------------------
    def test(self):
        with torch.no_grad():
            self.init_states()
            outputs, _ = self.rnn_model(self.inputs, self.states)
            outputs = outputs.to(self.device).detach()
            self.test_acc.append(self.get_accuracy(outputs, self.labels, self.batch_size))

    # ----------------------------------------------
    def get_test_acc(self):        
       
        self.test_acc = sum(self.test_acc)/len(self.test_acc) 
        
        print(f"Test accuracy is {self.test_acc}")
    # ----------------------------------------------

    def save_result(self):
        # Plot loss
        load_prefix = self.opt.load_iter if self.opt.load_iter > 0 else self.opt.epoch
        np.savez(self.result_dir + "/test_epoch_" + str(load_prefix) + "_T"+ str(self.T)+ ".npz", accuracy=self.test_acc)
   