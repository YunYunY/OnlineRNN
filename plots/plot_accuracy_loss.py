import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------
opt.taskid = 13

if opt.taskid == 0:
    d = "MNIST"
    # m = "StopBPRNN"
    m = "TBPTT"
elif opt.taskid == 1:
    d = "HAR_2"
    m = "VanillaRNN"
    total_batches = 2900#11600
elif opt.taskid == 2:
    d = "DSA_19"
    m = "VanillaRNN"
    total_batches = 7200 
elif opt.taskid == 3:
    d = "HAR_2"
    m = "TBPTT"
    optimizer = "FGSM_Adam"
    opt.iterT = 1
elif opt.taskid == 4:
    d = "HAR_2"
    m = "TBPTT"
    optimizer = "Adam"
    opt.iterT = 1
elif opt.taskid == 5:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 1
    optimizer = "irnn_Adam"
    total_batches = 20000
elif opt.taskid == 6:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 1
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 7:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 4
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 9:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 10
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 9:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 10
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 9:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 10
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 10:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 20
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 11:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 20
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 12:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 30
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 13:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 40
    optimizer = "FGSM_Adam"
    total_batches = 20000
elif opt.taskid == 14:
    d = "MNIST"
    m = "IndRNN"
    opt.iterT = 50
    optimizer = "FGSM_Adam"
    total_batches = 20000
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
# Plot training loss by epoch
# -----------------------------------------------------------------------------------------------
def plot_training_loss_epoch():
    total_epoch = 960
    epoches = range(0, total_epoch, 10)
    train_losses = []
    T = opt.iterT
    for epoch in epoches:
        loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
        # loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
        train_loss = np.load(loss_file)["losses"]
        train_losses.append(train_loss)
    print(max(train_losses))
    imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_trainloss.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("# Epoches")
    plt.ylabel("Training Loss")
    plt.title(f"{d} {m} {optimizer} niter={T}")
    # plt.title(f"{d} {m} {optimizer} niter={T}, max: %.2f"%max(test_acc))
    plt.xticks(range(1, total_epoch+1, 100), range(0, total_epoch, 100), rotation="vertical")
    plt.xlim(xmin=0, xmax=total_epoch)
    plt.ylim(ymin=0, ymax=1.2)
    plt.plot(epoches, train_losses, color='r')
    # plt.plot(epoches, test_acc, linestyle='--', marker='o', color='b')
    # plt.text(total_epoch-200, 90, 'Max: %.2f'%max(test_acc), fontsize=15)

    plt.savefig(img_dir + "/" + imgname)
    print(img_dir)
    print(imgname)

# -----------------------------------------------------------------------------------------------
# Plot accuracy change on testing data 
# -----------------------------------------------------------------------------------------------
def plot_test_acc():
    total_epoch = 960
    epoches = range(0, total_epoch, 10)
    test_acc = []
    T = opt.iterT
    for epoch in epoches:
        # acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        acc = np.load(acc_file)["test_acc"]
        test_acc.append(acc)
    print(max(test_acc))
    imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_testacc.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("# Epoches")
    plt.ylabel("Test Accuracy")
    plt.title(f"{d} {m} {optimizer} niter={T}, max: %.2f"%max(test_acc))
    plt.xticks(range(1, total_epoch+1, 100), range(0, total_epoch, 100), rotation="vertical")
    plt.xlim(xmin=0, xmax=total_epoch)
    plt.ylim(ymin=0, ymax=100)
    plt.plot(epoches, test_acc, color='b')
    # plt.plot(epoches, test_acc, linestyle='--', marker='o', color='b')
    # plt.text(total_epoch-200, 90, 'Max: %.2f'%max(test_acc), fontsize=15)

    plt.savefig(img_dir + "/" + imgname)
    print(img_dir)
    print(imgname)

# -----------------------------------------------------------------------------------------------
# Plot loss change in training by batches
# -----------------------------------------------------------------------------------------------
def plot_training_loss_batch():
 
    # batches = range(4, total_batches+1, 4)
    batches = range(30, total_batches+1, 30)

    nepoch = 100
    for T in [opt.iterT]:

        losses = []
        for ibatch in batches:
            loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/batch_" + str(ibatch) + "_losses.npz"
            losses.append(np.load(loss_file)["loss"])
        print(max(losses))
        # imgname = "/epoch_" + str(nepoch) + d + "_" + m + "_T" + str(T) + "_" + optimizer + "_losses.png"
        imgname = "/epoch_" + str(nepoch) + d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_losses.png"

        plt.figure(figsize=(8, 8))
        plt.xlabel("# Batches")
        plt.ylabel("Losses at W_k")
        plt.title(f"{d} Loss Change {m} {optimizer} niter={T}")
        plt.xticks(range(1, total_batches+1, 1000), range(0, total_batches, 1000), rotation="vertical")
        plt.xlim(xmin=0, xmax=total_batches)
        if d == "HAR_2":
            plt.ylim(ymin=0, ymax=0.4) # HAR
        elif d == "MNIST":
            plt.ylim(ymin=0, ymax=3.5)
        else:
            plt.ylim(ymin=0, ymax=3) # DSA


        plt.plot(batches, losses, color='r')
        plt.savefig(img_dir + "/" + imgname)

        # plt.figure(figsize=(8, 8))
        # plt.xlabel("Epoch")
        # plt.ylabel("Losses")
        # plt.plot(range(nepoch), losses, linestyle='--', marker='o', color='r')
        # plt.savefig(img_dir + "/" + imgname)
        print(img_dir)
        print(imgname)



plot_test_acc()
# plot_training_loss()
# plot_training_loss_batch()
# plot_training_loss_epoch()
