import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest" # saved model epoch name
nepoch = 10 # total epoch in training
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------
opt.taskid = 1

if opt.taskid == 0:
    d = "MNIST"
    # m = "StopBPRNN"
    m = "TBPTT"
elif opt.taskid == 1:
    d = "HAR_2"
    m = "VanillaRNN"
# optimizer = "Adam"
# optimizer = "FGSM"
optimizer = "SGD"
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Plot loss change in training task0
# -----------------------------------------------------------------------------------------------
def plot_training_loss():
    for T in opt.T:
        opt.T_ = T
        losses = []
        reg1s = []
        reg2s = []
        for iepoch in range(nepoch):
            loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/loss/" + str(iepoch) + "_losses.npz"
            losses.append(np.load(loss_file)["losses"])
            
            if m == "StopBPRNN":
                reg1s.append(np.load(loss_file)["reg1"])
                reg2s.append(np.load(loss_file)["reg2"])
        
        imgname = "/test_epoch_" + str(epoch) +"_" + d + "_" + m + "_losses.png"

        plt.figure(figsize=(8, 8))
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.plot(range(nepoch), losses, linestyle='--', marker='o', color='r')
        plt.savefig(img_dir + "/" + imgname)

        if m == "StopBPRNN":
            imgname = "/test_epoch_" + str(epoch) +"_" + d + "_" + m + "_reglosses.png"
            plt.figure(figsize=(8, 8))
            plt.plot(reg1s, label="Wih")
            plt.plot(reg2s, label="Whh")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(img_dir + "/" + imgname)
        break
            

# -----------------------------------------------------------------------------------------------
# Plot accuracy change on testing data task0
# -----------------------------------------------------------------------------------------------
def plot_test_acc():
    test_acc = []

    for T in opt.T:
        opt.T_ = T

        acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/test_epoch_" + str(epoch) + "_T" + str(T) + ".npz"
        print("-------------" + str(T) + "-------------")
        acc = np.load(acc_file)["accuracy"]
        print(acc)
        test_acc.append(acc)
    imgname = "/test_epoch_" + str(epoch) +"_" + d + "_" + m + "_accuracy.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("Truncation Parameter")
    plt.ylabel("Accuracy")
    plt.plot(opt.T, test_acc, linestyle='--', marker='o', color='r')
    plt.savefig(img_dir + "/" + imgname)


# -----------------------------------------------------------------------------------------------
# Plot loss change in training task1
# -----------------------------------------------------------------------------------------------
def plot_training_loss_task1():
    total_batches = 6000
    batches = range(4, total_batches+1, 4)

    nepoch = 100
    # for T in opt.T:
    for T in [4]:
        opt.iterT = T
        losses = []
        for ibatch in batches:
            loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss/batch_" + str(ibatch) + "_losses.npz"
            losses.append(np.load(loss_file)["loss"])
        print(len(losses))
        imgname = "/epoch_" + str(nepoch) + d + "_" + m + "_T" + str(T) + "_" + optimizer + "_losses.png"
        plt.figure(figsize=(8, 8))
        plt.xlabel("# Batches")
        plt.ylabel("Losses at W_k")
        plt.title(f"Loss Change {optimizer} niter={T}")
        plt.xticks(range(1, total_batches+1, 1000), range(0, total_batches, 1000), rotation="vertical")
        plt.xlim(xmin=0, xmax=total_batches)
        plt.plot(batches, losses, color='r')
        plt.savefig(img_dir + "/" + imgname)

        # plt.figure(figsize=(8, 8))
        # plt.xlabel("Epoch")
        # plt.ylabel("Losses")
        # plt.plot(range(nepoch), losses, linestyle='--', marker='o', color='r')
        # plt.savefig(img_dir + "/" + imgname)
        print(img_dir)

# plot_test_acc()
# plot_training_loss()
plot_training_loss_task1()

