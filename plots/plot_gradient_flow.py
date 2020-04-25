import os
import numpy as np
from onlinernn.options.train_options import TrainOptions
import matplotlib.pyplot as plt

opt = TrainOptions().parse()

# -----------------------------------------------------------------------------------------------
result_dir = "../result/"

# epoch = "latest" # saved model epoch name
# nepoch = 10 # total epoch in training
img_dir = os.path.join(result_dir, "final_plots/gradient_flow")
os.makedirs(img_dir, exist_ok=True)
print(img_dir)

# -----------------------------------------------------------------------------------------------
opt.taskid = 1
if opt.taskid == 0:
    d = "MNIST"
    # m = "StopBPRNN"
    m = "TBPTT"
elif opt.taskid == 1:
    d = "HAR_2"
    m = "VanillaRNN"
# optimizer = "Adam"
optimizer = "FGSM"

# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Plot gradient flow
# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
# -----------------------------------------------------------------------------------------------

def plot_gradient_flow():
    avg_gradient = []
    total_batches = 6000
    batches = range(1, total_batches+1, 10)
    # for T in opt.T:
    for T in [5]:
        print("-------------" + str(T) + "-------------")
        opt.T_ = T
        for ibatch in batches:
            gradient_file = os.path.join(result_dir, m, d, "T"+str(T)) + "/" + optimizer + "/" + str(ibatch) + "_weight_hh.npz"
            weight_hh_grad = np.load(gradient_file)["weight_hh"]
            avg_gradient.append(weight_hh_grad)
        imgname = d + "_" + m + "_T" + str(T) + "_" + optimizer + "_weight_hh_grad.png"

        plt.figure(figsize=(8, 8))
        plt.xlabel("# Batches")
        plt.ylabel("Average Gradient of W_hh")
        plt.title(f"Gradient Flow {optimizer} niter={T}")
        plt.xticks(range(1, total_batches+1, 1000), range(0, total_batches, 1000), rotation="vertical")
        plt.xlim(xmin=0, xmax=total_batches)

        plt.plot(batches, avg_gradient, color='r')
        plt.savefig(img_dir + "/" + imgname)


       # plt.plot(ave_grads, alpha=0.3, color="b")
        # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        # plt.xlim(xmin=0, xmax=len(ave_grads))
        # plt.grid(True)

plot_gradient_flow()

