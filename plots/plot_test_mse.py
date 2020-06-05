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
d = "ADDING"

taskids = [402, 606]

task_dic = {200: ["FGSM_Adam", 1, "LSTM"],
            201: ["Adam", 1, "LSTM"],
            904: ["FGSM_Adam", 1, "TBPTT"],
            905: ["FGSM_Adam", 1, "VanillaRNN"],
            606: ["FGSM_Adam", 1, "TBPTT"],
            402: ["Adam", 1, "TBPTT"]}



total_batch = 18000

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}


def plot_multi_batch():

    labels = ['SGD' , 'Ours', 'Ours norm K=10', 'Ours norm K=30']
    colors = ['black', 'crimson', 'c', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange']


    batches = range(0, total_batch, 100)
    

    imgname = d + "train_loss.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 18
    SIZE4 = 20

    plt.xlabel("# Batches", fontsize=SIZE4)
    plt.ylabel("Training MSE", fontsize=SIZE4)
    plt.title(f"Adding Task")

    plt.xticks(range(1, total_batch+1, 1000), range(0, total_batch, 1000), rotation="vertical", fontsize=SIZE3)
    plt.yticks(fontsize=SIZE3)

    plt.xlim(xmin=0, xmax=total_batch)
    plt.ylim(ymin=0., ymax=0.4)
    # plt.ylim(ymin=0.1, ymax=0.3)
    # plt.ylim(ymin=50, ymax=90)
    # plt.subplots_adjust(top = 0.99, bottom = 0.9, right = 1, left = 0.9, 
    #         hspace = 0., wspace = 0.)
 

    for i, taskid in enumerate(taskids):
        task = task_dic[taskid]
        optimizer = task[0]
        T = task[1]
        m = task[2]
            
        data = []
        for batch in batches:
            print(batch)
            # file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/batch_" + str(batch+1) + "_test_acc.npz"
            file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/batch_" + str(batch+1) + "_losses.npz"
            one_data = np.load(file)["loss"]

            data.append(one_data)
        print(max(data))
        plt.plot(batches, data, color=colors[i], label=labels[i])
    plt.legend(loc=2, prop={'size': SIZE3})

    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)



plot_multi_batch()