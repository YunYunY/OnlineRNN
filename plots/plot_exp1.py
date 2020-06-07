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
    task_dic = {10: ["VanillaRNN", "Adam", 1],
                611: ["VanillaRNN", "Adam", 1],
                612: ["VanillaRNN", "Adam", 1],
                613: ["VanillaRNN", "Adam", 1],
                11: ["TBPTT", "Adam", 1],
                13: ["VanillaRNN", "FGSM_Adam", 1],
                430: ["VanillaRNN", "FGSM_Adam", 1],
                113: ["TBPTT", "FGSM_Adam", 1],
                1130: ["TBPTT", "FGSM_Adam", 1],
                119: ["TBPTT", "FGSM_Adam", 1],
                114: ["TBPTT", "FGSM_Adam", 1],
                115: ["TBPTT", "FGSM_Adam", 1],
                116: ["TBPTT", "FGSM_Adam", 1],
                117: ["TBPTT", "FGSM_Adam", 1],
                118: ["TBPTT", "FGSM_Adam", 1],
                1190: ["TBPTT", "FGSM_Adam", 1],
                212: ["VanillaRNN", "FGSM_Adam", 5],
                400: ["VanillaRNN", "FGSM_Adam", 5],
                1212: ["VanillaRNN", "FGSM_Adam", 5],   
                433: ["VanillaRNN", "FGSM_Adam", 5],   
                2400: ["VanillaRNN", "FGSM_Adam", 5],               
                1412: ["VanillaRNN", "FGSM_Adam", 30],
                1213: ["TBPTT", "FGSM_Adam", 5],
                1214: ["TBPTT", "FGSM_Adam", 5],
                1215: ["TBPTT", "FGSM_Adam", 5],
                1217: ["TBPTT", "FGSM_Adam", 5],
                1433: ["TBPTT", "FGSM_Adam", 5],
                1216: ["TBPTT", "FGSM_Adam", 5],
                219: ["TBPTT", "FGSM_Adam", 5],
                2190: ["TBPTT", "FGSM_Adam", 5],
                }

else:
    task_dic = {0: ["VanillaRNN", "Adam", 1],
                15: ["TBPTT", "Adam", 1],
                2: ["VanillaRNN", "FGSM_Adam", 1],
                3: ["VanillaRNN", "FGSM_Adam", 1],
                1002: ["VanillaRNN", "FGSM_Adam", 1],
                2021: ["TBPTT", "FGSM_Adam", 1],
                2022: ["TBPTT", "FGSM_Adam", 1],
                202: ["VanillaRNN", "FGSM_Adam", 5],
                205: ["VanillaRNN", "FGSM_Adam", 5],
                204: ["VanillaRNN", "FGSM_Adam", 5],
                1202: ["VanillaRNN", "FGSM_Adam", 5],
                2014: ["TBPTT", "FGSM_Adam", 5],
                2015: ["TBPTT", "FGSM_Adam", 5]}

plot_by = "batch"

if plot_by == "batch":
    # plot by batches
    total_batches = 80000 #12000
    n_update_ = 16000-1
    scale = 100 # every scale update read one data 
    n_update = int(n_update_/scale)
else:
    # plot by epoch
    total_epoch = 180 # 479

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------

def plot_multi_epoch():
    case = "losses" # "test_acc" # "losses"
    if dname == "PermuteMNIST":
        # taskids = [613, 11, 430, 115, 212] 433
        taskids = [613, 11, 430, 1190, 1212, 1217]

    else:
        # taskids = [0, 15, 1002, 2021, 1202, 2014]
        # taskids = [0, 15, 2, 2021, 202, 2014]
        taskids = [0, 15, 3, 2022, 204, 2015]

    # colors = ['c', 'crimson', 'darkorange', 'mediumblue', 'springgreen']
    if dname == "PermuteMNIST":
        labels = ['SGD', 'TBPTT', 'Ours K=1', 'TBPTT+Ours K=1', 'Ours K=5', 'TBPTT+Ours K=5']
    else:
        labels = ['SGD', 'TBPTT', 'Ours K=1', 'TBPTT+Ours K=1', 'Ours K=5', 'TBPTT+Ours K=5']

    colors = ['black', 'deepskyblue', 'crimson', 'darkorange', 'mediumblue', 'springgreen']

    # colors = ['black', 'c', 'crimson', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange', 'springgreen]
    case_dir = {"losses": "Training Loss", 
                "test_acc": "Test Accuracy"}

    if plot_by != "batch":
        epochs = range(0, total_epoch, 1)

    str_ids = ''.join(map(str, taskids))
    imgname = "P1_" + d + "_" + str_ids + "_"+ case +".pdf"
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
        plt.title("Vanilla RNN, Permute-MNIST, L=784", fontsize=SIZE4)
    else:
        plt.title("Vanilla RNN, Pixel-MNIST, L=784", fontsize=SIZE4)

    if case == "losses":
        if plot_by == "batch":
            plt.xticks(range(1, int(n_update_+1), 10), range(0, int(n_update_), 10), rotation="vertical", fontsize=SIZE3)
        else:
            plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)

    else:
        plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)

    plt.yticks(fontsize=SIZE3)

    if plot_by == "batch":
        plt.xlim(xmin=0, xmax=n_update)
    else:
        plt.xlim(xmin=0, xmax=total_epoch)

    if case == "losses":
        if plot_by == "batch":
            plt.ylim(ymin=0, ymax=3)
        else:
            plt.ylim(ymin=0., ymax=3)

    else:
        if dname == "PermuteMNIST":
            plt.ylim(ymin=20, ymax=90)
        else:
            plt.ylim(ymin=0, ymax=100)

 

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
                # xrange = [x for x in range(0, scale*len(epochs), scale)]
        
        data = []
        print(epochs[0:10])
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
        if plot_by == "batch":
            if dname == "PermuteMNIST":
                plt.legend(loc=1, prop={'size': SIZE3})
            else:
                plt.legend(loc=3, prop={'size': SIZE3})
        else:
            plt.legend(loc=1, prop={'size': SIZE3})
    else:
        if dname == "PermuteMNIST":
            plt.legend(loc=4, prop={'size': SIZE3})
        else:
            plt.legend(loc=2, prop={'size': SIZE3})


    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)


plot_multi_epoch()