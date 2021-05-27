import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------
baseline = True # plot with baseline models

if baseline:

    # baselines = ['fastrnn', 'lipchiz', 'irnn', 'momentumrnn', 'rnn']
    baselines = ['fastrnn', 'irnn', 'lipchiz', 'rnn']

    result_baseline = result_dir + "baseline/"

    def load_baseline(dir, test_acc):
        with open(result_baseline + 'har_result_' + str(dir) + '.pkl', 'rb') as f:
            Bdata = pickle.load(f)
            
            Btrain_loss = Bdata['train_loss']
            Btest_acc = Bdata['test_acc']
            print(Btest_acc)
            exit(0)
        if test_acc:
            return Btest_acc
        else:
            return Btrain_loss

load_baseline('fastrnn', True)

d = "HAR_2"
dname = "HAR_2"
# dname = "NoisyHAR_2"

if dname == "HAR_2":
    task_dic = {1027: ["VanillaRNN", "Adam", 1],
                102: ["VanillaRNN", "Adam", 1], 
                101: ["VanillaRNN", "Adam", 1], 
                104: ["VanillaRNN", "Adam", 1], 
                110: ["VanillaRNN", "Adam", 1], 
                2027: ["VanillaRNN", "Adam", 1], 
                3027: ["VanillaRNN", "Adam", 1], 
                206: ["VanillaRNN", "Adam", 2], 
                214: ["VanillaRNN", "Adam", 2], 
                304: ["VanillaRNN", "Adam", 1],
                305: ["VanillaRNN", "Adam", 2],
                306: ["VanillaRNN", "Adam", 1],
                314: ["VanillaRNN", "Adam", 1],}
else:
    task_dic = {20: ["VanillaRNN", "Adam", 1],
                21: ["TBPTT", "Adam", 1], 
                22: ["VanillaRNN", "FGSM_Adam", 1], 
                23: ["VanillaRNN", "FGSM_Adam", 5], 
                24: ["VanillaRNN", "FGSM_Adam", 10],
                25: ["TBPTT", "FGSM_Adam", 1],
                26: ["TBPTT", "FGSM_Adam", 10]}

plot_by = "epoch"
if plot_by == "batch":
    total_batches = 11430 #11600
    n_update_ = 1143-1
    scale = 10 # every scale update read one data 
    n_update = int(n_update_/scale)
else:
    total_epoch = 160 # 199

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy", 
            "train_acc": "Train Accuracy"}

def plot_multi_epoch():
    case = "losses" # "test_acc" # "losses"
    if dname == "HAR_2":
        # taskids = [100, 102, 203, 205, 304, 305]
        taskids = [1027, 2027, 3027]
        # taskids = [105, 207, 307]

    else:
        taskids = [20, 21, 22, 23, 24, 26]

    labels = ['SGD', 'SHB', 'SNAG', 'HB K=S=1', 'HB K=S=2', 'NAG K=S=1', 'HB K=S=2']
    colors = ['black', 'deepskyblue', 'lightsteelblue', 'violet', 'crimson', 'darkorange']

    # colors = ['black', 'c', 'lightsteelblue', 'violet', 'crimson', 'darkorange']

    if plot_by != "batch":
        epochs = range(0, total_epoch, 1)

    str_ids = ''.join(map(str, taskids))
    imgname = d + "_" + str_ids + "_"+ case +".png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 18
    SIZE4 = 20
    if plot_by == "batch":
        # after times same as scale
        plt.xlabel(r"# Updates ($1\times{10^1}$)", fontsize=SIZE4)
    else:
        plt.xlabel("# Epochs", fontsize=SIZE4)
    plt.ylabel(case_dir[case], fontsize=SIZE4)

    if dname == "HAR_2":
        plt.title("RNN, HAR-2, L=128, 1%", fontsize=SIZE4)
    else:
        plt.title("Vanilla RNN, Noisy HAR-2, L=128", fontsize=SIZE4)

    if plot_by == "batch":
        plt.xticks(range(1, int(n_update_+1), 10), range(0, int(n_update_), 10), rotation="vertical", fontsize=SIZE3)
    else:
        plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)
    plt.yticks(fontsize=SIZE3)

    if plot_by == "batch":
        plt.xlim(xmin=0, xmax=n_update)
    else:
        plt.xlim(xmin=0, xmax=total_epoch)

    if dname == "HAR_2":
        if case == "losses":
            plt.ylim(ymin=0., ymax=0.4)
        else:
            plt.ylim(ymin=50, ymax=100)

    else:
        plt.ylim(ymin=0.05, ymax=0.4)
    # plt.ylim(ymin=50, ymax=90)

  

    for i, taskid in enumerate(taskids):
        m = task_dic[taskid][0]
        optimizer = task_dic[taskid][1]
        T = task_dic[taskid][2]

              
        if plot_by == "batch":
            if case == "losses":
                epoch = T
                epochs = []
                while epoch < total_batches:
                    epochs.append(epoch)
                    epoch += scale* T

        data = []
        for epoch in epochs:
            if case == "losses":
                if plot_by == "batch":
                    file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/batch_" + str(epoch) + "_losses.npz"
                    one_data = np.load(file)["loss"]
                else:
                    file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
                    one_data = np.load(file)[case]

            elif case == "test_acc":
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
                one_data = np.load(file)[case]
            elif case == "train_acc":
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
                one_data = np.load(file)[case]

            data.append(one_data)
        print(max(data))
        if plot_by == "batch":
            xrange = [x for x in range(0, n_update, 1)]
            plt.plot(xrange, data[0:n_update], color=colors[i], label=labels[i], linewidth=2)
        else:
            plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)

    if case == "losses":
        if plot_by == "batch":
            plt.legend(loc=1, prop={'size': SIZE3})
        else:
            plt.legend(loc=2, prop={'size': SIZE3})
    else:
        plt.legend(loc=4, prop={'size': SIZE3})


    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)



# -----------------------------------------------------------------------------------------------
# Plot accuracy change on testing data 
# -----------------------------------------------------------------------------------------------
def plot_multi_test_acc():
    epochs = range(0, total_epoch, 1)
    # epochs = range(0, total_epoch, 10)
    test_acc = []
    T = opt.iterT
    for epoch in epochs:
        # acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        test_acc.append(acc)
    print(max(test_acc))
    imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_testacc.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("# Epochs")
    plt.ylabel("Test Accuracy")
    plt.title(f"{d} {m} {optimizer} niter={T}, max: %.2f"%max(test_acc))
    # plt.xticks(range(1, total_epoch+1, 100), range(0, total_epoch, 100), rotation="vertical")
    plt.xticks(range(1, total_epoch+1, 10), range(0, total_epoch, 10), rotation="vertical")
    plt.xlim(xmin=0, xmax=total_epoch)
    plt.ylim(ymin=0, ymax=100)
    plt.plot(epochs, test_acc, color='b')
    # plt.plot(epochs, test_acc, linestyle='--', marker='o', color='b')
    # plt.text(total_epoch-200, 90, 'Max: %.2f'%max(test_acc), fontsize=15)

    plt.savefig(img_dir + "/" + imgname)
    print(img_dir)
    print(imgname)


def plot_training_loss_epoch():
    total_epoch = 49 #960
    epochs = range(0, total_epoch, 1)
    train_losses = []
    T = opt.iterT
    for epoch in epochs:
        loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
        # loss_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
        train_loss = np.load(loss_file)["losses"]
        train_losses.append(train_loss)
    print(max(train_losses))
    imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_trainloss.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("# Epochs")
    plt.ylabel("Training Loss")
    plt.title(f"{d} {m} {optimizer} niter={T}")
    # plt.title(f"{d} {m} {optimizer} niter={T}, max: %.2f"%max(test_acc))
    plt.xticks(range(1, total_epoch+1, 10), range(0, total_epoch, 10), rotation="vertical")
    plt.xlim(xmin=0, xmax=total_epoch)
    # plt.ylim(ymin=0, ymax=1.2)
    plt.ylim(ymin=0, ymax=0.2)
    plt.plot(epochs, train_losses, color='r')
    # plt.plot(epochs, test_acc, linestyle='--', marker='o', color='b')
    # plt.text(total_epoch-200, 90, 'Max: %.2f'%max(test_acc), fontsize=15)

    plt.savefig(img_dir + "/" + imgname)
    print(img_dir)
    print(imgname)

# -----------------------------------------------------------------------------------------------
# Plot accuracy change on testing data 
# -----------------------------------------------------------------------------------------------
def plot_test_acc():
    total_epoch = 49 #960
    epochs = range(0, total_epoch, 1)
    # epochs = range(0, total_epoch, 10)
    test_acc = []
    T = opt.iterT
    for epoch in epochs:
        # acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(opt.taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
        acc = np.load(acc_file)["test_acc"]
        test_acc.append(acc)
    print(max(test_acc))
    imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + str(opt.taskid)+ "_testacc.png"

    plt.figure(figsize=(8, 8))
    plt.xlabel("# Epochs")
    plt.ylabel("Test Accuracy")
    plt.title(f"{d} {m} {optimizer} niter={T}, max: %.2f"%max(test_acc))
    # plt.xticks(range(1, total_epoch+1, 100), range(0, total_epoch, 100), rotation="vertical")
    plt.xticks(range(1, total_epoch+1, 10), range(0, total_epoch, 10), rotation="vertical")
    plt.xlim(xmin=0, xmax=total_epoch)
    plt.ylim(ymin=0, ymax=100)
    plt.plot(epochs, test_acc, color='b')
    # plt.plot(epochs, test_acc, linestyle='--', marker='o', color='b')
    # plt.text(total_epoch-200, 90, 'Max: %.2f'%max(test_acc), fontsize=15)

    plt.savefig(img_dir + "/" + imgname)
    print(img_dir)
    print(imgname)

# -----------------------------------------------------------------------------------------------
# Plot loss change in training by batches
# -----------------------------------------------------------------------------------------------
def plot_training_loss_batch():
 
    batches = range(4, total_batches+1, 4)
    # batches = range(30, total_batches+1, 30)

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



# plot_test_acc()
# plot_training_loss()
# plot_training_loss_batch()
# plot_training_loss_epoch()
plot_multi_epoch()