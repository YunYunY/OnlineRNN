import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)


# -----------------------------------------------------------------------------------------------
d = "ADDING"

baseline = True # plot with baseline models
if baseline:

    # baselines = ['fastrnn', 'lipchiz', 'irnn', 'momentumrnn', 'rnn']
    baselines = ['fastrnn', 'lipchiz', 'rnn']

    result_baseline = result_dir + "baseline/"

    def load_baseline(dir):
        with open(result_baseline + 'result_' + str(dir) + '_' + str(seq) +'.pkl', 'rb') as f:
            Bdata = pickle.load(f)
            Btrain_loss = Bdata['train_loss']
            # Btest_loss = Bdata['test_loss']
        return Btrain_loss

  
seq = 100

# 100
taskids = [5031, 5041, 6031, 6041, 7031, 7041]
# 200
# taskids = [5032, 5042, 6032, 6042, 7032, 7042]
# 500
# taskids = [5033, 5043, 6033, 6043, 7033, 7043]
# 1000
# taskids = []
nexp = len(taskids)

# taskids = []
 
task_dic = {5031: ["Adam", 1, "VanillaRNN"],
            5041: ["Adam", 1, "VanillaRNN"],
            6031: ["Adam", 1, "VanillaRNN"],
            6041: ["Adam", 1, "VanillaRNN"],
            7031: ["Adam", 1, "VanillaRNN"],
            7041: ["Adam", 1, "VanillaRNN"],
            3151: ["Adam", 1, "VanillaRNN"]}

# total_batch = 100000
total_batch = 25000

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}


def plot_multi_batch():
    if baseline:
        cindex = 0 
    # labels = ['SGDrelu', 'HBrelu', 'NAGrelu']

    # labels = ['FastRNN', 'LipschitzRNN']
    # labels = ['dense SGD', 'sparse SGD', 'dense SHB', 'sparse SHB', 'dense SNAG', 'sparse SNAG',\
    #      'FastRNN', 'LipschitzRNN', 'iRNN', 'MomentumRNN', 'VanillaRNN']
    labels = ['dense SGD', 'sparse SGD', 'dense SHB', 'sparse SHB', 'dense SNAG', 'sparse SNAG',\
         'FastRNN', 'LipschitzRNN', 'VanillaRNN']
    # labels = ['HB', 'NAG']
    # labels = ['GD', 'HB \mu 0.5', 'NAG \mu 0.5', 'HB \mu 0.99', 'NAG \mu 0.99', 'HB \mu 0.048', 'NAG \mu 0.3']
    # colors = ['black', 'crimson',  'darkorange', 'mediumblue', 'violet', 'c', 'violet', ]
    colors = ['black', 'crimson', 'deepskyblue' , 'darkorange', 'violet', 'mediumblue', 'blueviolet', 'c', 'lightsteelblue', 'dimgray', 'silver', "red", 'slateblue']


    # batches = range(9, total_batch, 10)
    batches = range(9, total_batch, 500)

    
    str_ids = ''.join(map(str, taskids))
    # imgname = 'Addingtest.png'
    imgname = "Adding_" + str(seq) + ".pdf"
    # imgname = "Adding" + d + "_" + str_ids + "_.pdf"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 18
    SIZE4 = 20

    plt.xlabel(r"# Training Steps ($5\times{10^3}$)", fontsize=SIZE4)
    plt.ylabel("Training MSE", fontsize=SIZE4)
    plt.title("Adding Task, T=" + str(seq), fontsize=SIZE4)
    # plt.title("meta-RNN-dense, Adding Task, L=100", fontsize=SIZE4)
    plt.xticks(range(1, total_batch+1, 5000), range(0, int(total_batch/5000), 1), rotation="vertical", fontsize=SIZE3)

    # plt.xticks(range(1, total_batch+1, 5000), range(0, total_batch, 5000), rotation="vertical", fontsize=SIZE3)
    plt.yticks(fontsize=SIZE3)

    plt.xlim(xmin=0, xmax=total_batch)
    plt.ylim(ymin=0., ymax=0.3)
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
     
        if baseline:
            if i < nexp and i%2 == 1:
                plt.plot(batches, data, color=colors[cindex], label=labels[i], linestyle='dashed', linewidth=4)
                cindex += 1
            else: 
                plt.plot(batches, data, color=colors[cindex], label=labels[i], linewidth=4)
        else:
            plt.plot(batches, data, color=colors[i], label=labels[i])

    if baseline:
        i = i + 1
        for _, b in enumerate(baselines):
       
            # fast_loss = load_baseline('fastrnn')
            # Lipschitz_loss = load_baseline('lipchiz')
            b_loss = load_baseline(b)
            data = []
            for batch in batches:
                print(batch)
                data.append(b_loss[batch])
            plt.plot(batches, data, color=colors[cindex], label=labels[i])
            i += 1
            cindex += 1
    plt.legend(loc=1, prop={'size': SIZE3})

    plt.savefig(img_dir + "/" + imgname, bbox_inches='tight')
    print(img_dir)
    print(imgname)



plot_multi_batch()