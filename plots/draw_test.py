import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "plots", epoch)
os.makedirs(img_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------

if opt.taskid == 0:
    d = "MNISTShift"
    m = "StopBPRNN"

# -----------------------------------------------------------------------------------------------

test_acc = []

for T in opt.T:
    opt.T_ = T

    acc_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/test_epoch_" + str(epoch) + "_T" + str(T) + ".npz"
    print("-------------" + str(T) + "-------------")
    acc = np.load(acc_file)["accuracy"]
    print(acc)
    test_acc.append(acc)
imgname = "/test_epoch_" + str(epoch) +"_" + d + "_" + m + "_accuracy.png"

# -----------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 8))
plt.xlabel("Truncation Parameter")
plt.ylabel("Accuracy")
plt.plot(opt.T, test_acc, linestyle='--', marker='o', color='r')
plt.savefig(img_dir + "/" + imgname)


