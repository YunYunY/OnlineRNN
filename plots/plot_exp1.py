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

# total_batches = 12000
# total_epoch = 12000

total_epoch = 180 # 479

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}

def plot_multi_epoch():
    case = "test_acc" # "test_acc" # "losses"
   
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

    colors = ['black', 'c', 'crimson', 'darkorange', 'mediumblue', 'springgreen']

    # colors = ['black', 'c', 'crimson', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange', 'springgreen]


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
        # plt.xlabel(r"# Training steps ($1\times{10^3}$)", fontsize=SIZE4)
        plt.xlabel("# Epochs", fontsize=SIZE4)
    else:
        plt.xlabel("# Epochs", fontsize=SIZE4)
    plt.ylabel(case_dir[case], fontsize=SIZE4)
    # plt.title(f"{d}", fontsize=SIZE4)
    if dname == "PermuteMNIST":
        plt.title("Vanilla RNN, Permute-MNIST, L=784", fontsize=SIZE4)
    else:
        plt.title("Vanilla RNN, Pixel-MNIST, L=784", fontsize=SIZE4)
    # plt.title("Noisy HAR-2 L=128", fontsize=SIZE4)

    if case == "losses":
        # plt.xticks(range(1, total_epoch+1, 1000), range(0, int(total_epoch/1000), 1), rotation="vertical", fontsize=SIZE3)
        plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)

    else:
        plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)

    plt.yticks(fontsize=SIZE3)

    plt.xlim(xmin=0, xmax=total_epoch)
    if case == "losses":
        # plt.ylim(ymin=1.5, ymax=3)
        plt.ylim(ymin=0., ymax=3)

    else:
        if dname == "PermuteMNIST":
            plt.ylim(ymin=20, ymax=90)
        else:
            plt.ylim(ymin=0, ymax=100)

    # plt.subplots_adjust(top = 0.99, bottom = 0.9, right = 1, left = 0.9, 
    #         hspace = 0., wspace = 0.)
 

    for i, taskid in enumerate(taskids):

        m = task_dic[taskid][0]
        optimizer = task_dic[taskid][1]
        T = task_dic[taskid][2]
  
        # scale = 100
        # if case == "losses":
        #     epoch = T
        #     epochs = []
        #     while epoch < total_batches:
        #         epochs.append(epoch)
        #         epoch += scale* T
        #     xrange = [x for x in range(0, scale*len(epochs), scale)]
        
                  
        data = []
        for epoch in epochs:
            if case == "losses":
                # file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/batch_" + str(epoch) + "_losses.npz"
                # one_data = np.load(file)["loss"]
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
                one_data = np.load(file)[case]
            elif case == "test_acc":
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
                one_data = np.load(file)[case]

            data.append(one_data)
        print(max(data))
        if case == "losses":
            # plt.plot(xrange, data, color=colors[i], label=labels[i], linewidth=2)
            plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)
        else:
            plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)

    if case == "losses":
        plt.legend(loc=1, prop={'size': SIZE3})
    else:
        if dname == "PermuteMNIST":
            plt.legend(loc=4, prop={'size': SIZE3})
        else:
            plt.legend(loc=2, prop={'size': SIZE3})


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