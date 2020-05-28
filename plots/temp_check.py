import os
import numpy as np
import math
from statistics import mean, stdev

result_dir = "../result/"
d = "HAR_2"
m = "VanillaRNN"
taskid = "7"
T = "10"
optimizer = "FGSM_Adam"
file = os.path.join("../result", m, d, "T"+T, optimizer, taskid, "loss_acctemp_dis.npz")
dis = np.load(file)["dis"]
degree = [math.degrees(math.acos(x)) for x in dis]
print(mean(degree))
print(stdev(degree))
