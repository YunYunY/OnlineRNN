import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------
d = "MNIST"
dname = "PermuteMNIST"
# dname = "MNIST"


if dname == "PermuteMNIST":
    task_dic = {325: ["IndRNN", "irnn_Adam", 1],
                326: ["IndRNN", "FGSM_Adam", 1], 
                328: ["IndRNN", "FGSM_Adam", 5],
                410: ["IndRNN", "FGSM_Adam", 10],
                510: ["IndRNN", "FGSM_Adam", 30],
                32500: ["IndRNN", "irnn_Adam", 1],
                32600: ["IndRNN", "FGSM_Adam", 1], 
                32800: ["IndRNN", "FGSM_Adam", 5],
                41000: ["IndRNN", "FGSM_Adam", 10],
                51000: ["IndRNN", "FGSM_Adam", 30]}
else:
    task_dic = {25: ["IndRNN", "irnn_Adam", 1],
                29: ["IndRNN", "FGSM_Adam", 1], 
                30: ["IndRNN", "FGSM_Adam", 5],
                27: ["IndRNN", "FGSM_Adam", 10],
                28: ["IndRNN", "FGSM_Adam", 30],
                2500: ["IndRNN", "irnn_Adam", 1],
                2900: ["IndRNN", "FGSM_Adam", 1], 
                3000: ["IndRNN", "FGSM_Adam", 5],
                3100: ["IndRNN", "FGSM_Adam", 10],
                2800: ["IndRNN", "FGSM_Adam", 30]
                }

plot_by = "batch"
if plot_by == "batch":
    total_batches = 187031 #12000
    n_update_ = 37406-1
    scale = 100 # every scale update read one data 
    n_update = int(n_update_/scale)
else:
    total_epoch = 399

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}

def plot_multi_epoch():
    case = "losses" # "test_acc" # "losses"
    if dname == "PermuteMNIST":
        # taskids = [325, 326, 328, 410, 510]
        # taskids = [32500, 32600, 32800, 41000, 51000]
        taskids = [32500, 32600, 32800]


    else:
        # taskids = [25, 29, 30, 27, 28]
        # taskids = [2500, 2900, 3000, 3100, 2800]
        taskids = [2500, 2900, 3000]



    # labels = ['TBPTT', 'Ours K=1', 'TBPTT+Ours K=1', 'Ours K=5', 'TBPTT+Ours K=5']
    # colors = ['c', 'lightsteelblue', 'darkorange', 'mediumblue', 'springgreen']

    labels = ['SGD', 'Ours K=1', 'Ours K=5', 'Ours K=10', 'Ours K=30']
    colors = ['black', 'violet', 'crimson', 'darkorange', 'springgreen']

    # colors = ['black', 'c', 'crimson', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange', 'springgreen]

    if plot_by != "batch":
        epochs = range(0, total_epoch, 1)

    str_ids = ''.join(map(str, taskids))
    imgname = "P2_" + d + "_" + str_ids + "_"+ case +".pdf"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 18
    SIZE4 = 20
    if case == "losses": 
        if plot_by == "batch":
            # after times same as scale
            plt.xlabel(r"# Updates ($1\times{10^2}$)", fontsize=SIZE4)
        else:
            plt.xlabel("# Epochs", fontsize=SIZE4)
    else:
        plt.xlabel("# Epochs", fontsize=SIZE4)
    plt.ylabel(case_dir[case], fontsize=SIZE4)

    if dname == "PermuteMNIST":
        plt.title("IndRNN, Permute-MNIST L=784", fontsize=SIZE4)
    else:
        plt.title("IndRNN, Pixel-MNIST, L=784", fontsize=SIZE4)

    if case == "losses":
        if plot_by == "batch":
            plt.xticks(range(1, int(n_update_+1), 20), range(0, int(n_update_), 20), rotation="vertical", fontsize=SIZE3)
        else:
            plt.xticks(range(1, total_epoch+1, 40), range(0, total_epoch, 40), rotation="vertical", fontsize=SIZE3)

    else:
        plt.xticks(range(1, total_epoch+1, 40), range(0, total_epoch, 40), rotation="vertical", fontsize=SIZE3)

    plt.yticks(fontsize=SIZE3)

    if plot_by == "batch":
        plt.xlim(xmin=0, xmax=n_update)
    else:
        plt.xlim(xmin=0, xmax=total_epoch)
    if case == "losses":
        if dname == "PermuteMNIST":
            plt.ylim(ymin=0, ymax=1.5)
        else:
            plt.ylim(ymin=0, ymax=1)
    else:
        if dname == "PermuteMNIST":
            plt.ylim(ymin=80, ymax=94)
        else:
            plt.ylim(ymin=90, ymax=100)



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

            data.append(one_data)
        print(max(data))
        if case == "losses":
            if plot_by == "batch":
                xrange = [x for x in range(0, n_update, 1)]
                plt.plot(xrange, data[0:n_update], color=colors[i], label=labels[i], linewidth=2)
            else:
                plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)
        else:
            plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)

    if case == "losses":
        plt.legend(loc=1, prop={'size': SIZE3})
    else:
        plt.legend(loc=4, prop={'size': SIZE3})


    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)



plot_multi_epoch()