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
opt.taskid = 4
d = "HAR_2"
task_dic = {0: ["VanillaRNN", "Adam", 1],
            1: ["TBPTT", "Adam", 1], 
            2: ["VanillaRNN", "FGSM_Adam", 1], 
            3: ["VanillaRNN", "FGSM_Adam", 5], 
            4: ["VanillaRNN", "FGSM_Adam", 10],
            5: ["TBPTT", "FGSM_Adam", 1]}

# task_dic = {20: ["VanillaRNN", "Adam", 1],
#             21: ["TBPTT", "Adam", 1], 
#             22: ["VanillaRNN", "FGSM_Adam", 1], 
#             23: ["VanillaRNN", "FGSM_Adam", 5], 
#             24: ["VanillaRNN", "FGSM_Adam", 10],
#             25: ["TBPTT", "FGSM_Adam", 1],
#             26: ["TBPTT", "FGSM_Adam", 10]}

total_batches = 2900#11600
total_epoch = 199

# -----------------------------------------------------------------------------------------------
# Plot multiple training loss by epoch in one
# -----------------------------------------------------------------------------------------------
case_dir = {"losses": "Training Loss", 
            "test_acc": "Test Accuracy"}

def plot_multi_epoch():
    case = "test_acc" # "test_acc" # "losses"
    taskids = [0, 1, 2, 3, 4, 5]
    # taskids = [20, 21, 22, 23, 24, 26]

    labels = ['SGD', 'TBPTT', 'Ours K=1', 'Ours K=5', 'Ours K=10', 'TBPTT+Ours']
    colors = ['black', 'c', 'lightsteelblue', 'violet', 'crimson', 'darkorange']

    # colors = ['black', 'c', 'crimson', 'violet', 'mediumblue', 'lightsteelblue', 'darkorange']


    epochs = range(0, total_epoch, 1)

    str_ids = ''.join(map(str, taskids))
    imgname = d + "_" + str_ids + "_"+ case +".pdf"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    SIZE1 = 8
    SIZE2 = 10
    SIZE3 = 16
    SIZE4 = 18

    plt.xlabel("# Epochs", fontsize=SIZE4)
    plt.ylabel(case_dir[case], fontsize=SIZE4)
    # plt.title(f"{d}", fontsize=SIZE4)
    plt.title("HAR-2 L=128", fontsize=SIZE4)
    # plt.title("Noisy HAR-2 L=128", fontsize=SIZE4)


    plt.xticks(range(1, total_epoch+1, 20), range(0, total_epoch, 20), rotation="vertical", fontsize=SIZE3)
    plt.yticks(fontsize=SIZE3)

    plt.xlim(xmin=0, xmax=total_epoch)
    # plt.ylim(ymin=0., ymax=0.4)
    # plt.ylim(ymin=0.05, ymax=0.35)
    # plt.ylim(ymin=50, ymax=90)
    plt.ylim(ymin=50, ymax=95)

    # plt.subplots_adjust(top = 0.99, bottom = 0.9, right = 1, left = 0.9, 
    #         hspace = 0., wspace = 0.)
 

    for i, taskid in enumerate(taskids):
        m = task_dic[taskid][0]
        optimizer = task_dic[taskid][1]
        T = task_dic[taskid][2]
            
        data = []
        for epoch in epochs:
            if case == "losses":
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_losses_train_acc.npz"
            elif case == "test_acc":
                file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(taskid) + "/loss_acc/epoch_" + str(epoch) + "_test_acc.npz"
            one_data = np.load(file)[case]

            data.append(one_data)
        print(max(data))
        plt.plot(epochs, data, color=colors[i], label=labels[i], linewidth=2)
    if case == "losses":
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