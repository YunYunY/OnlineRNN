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
# m = "VanillaRNN"
# m = "TBPTT"

task_dic = {0: [100, "Adam", 1, "VanillaRNN"],
            1: [100, "Adam", 1, "TBPTT"],
            2: [200, "FGSM_Adam", 1, "TBPTT"]}
# task_dic = {0: [100, "Adam", 1],
#             1: [200, "FGSM_Adam", 1], 
#             2: [200, "FGSM_Adam", 5], 
#             3: [200, "FGSM_Adam", 10],
#             4: [200, "FGSM_Adam", 30]}


total_batch = 600

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}

def plot_multi_batch():

    labels = ['Adam RNN', 'TBPTT', 'TBPTT Ours norm K=1', 'Ours norm K=5', 'Ours norm K=10', 'Ours norm K=30']
    colors = ['black', 'c', 'crimson', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange']


    batches = range(0, total_batch, 1)

    imgname = d + "_test_acc.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 12
    SIZE4 = 14

    plt.xlabel("# Batches", fontsize=SIZE4)
    plt.ylabel("Test MSE", fontsize=SIZE4)
    plt.title(f"{d}", fontsize=SIZE4)
    # plt.title(f"Noisy HAR-2")

    plt.xticks(range(1, total_batch+1, 100), range(0, total_batch, 100), rotation="vertical", fontsize=SIZE3)
    plt.yticks(fontsize=SIZE3)

    plt.xlim(xmin=0, xmax=total_batch)
    plt.ylim(ymin=0., ymax=0.4)
    # plt.ylim(ymin=0.1, ymax=0.3)
    # plt.ylim(ymin=50, ymax=90)
    # plt.subplots_adjust(top = 0.99, bottom = 0.9, right = 1, left = 0.9, 
    #         hspace = 0., wspace = 0.)
 

    for i, task in task_dic.items():
        print(i)
        taskid = task[0]
        optimizer = task[1]
        T = task[2]
        m = task[3]
            
        data = []
        for batch in batches:
            file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/batch_" + str(batch+1) + "_test_acc.npz"
            one_data = np.load(file)["test_acc"]

            data.append(one_data)
        print(max(data))
        plt.plot(batches, data, color=colors[i], label=labels[i])
    plt.legend(loc=2, prop={'size': SIZE3})

    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)



plot_multi_batch()