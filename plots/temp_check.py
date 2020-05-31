import os
import numpy as np
import math
from statistics import mean, stdev

# result_dir = "../result/"
# d = "HAR_2"
# m = "VanillaRNN"
# taskid = "7"
# T = "10"
# optimizer = "FGSM_Adam"
# file = os.path.join("../result", m, d, "T"+T, optimizer, taskid, "loss_acctemp_dis.npz")
# dis = np.load(file)["dis"]
# degree = [math.degrees(math.acos(x)) for x in dis]
# print(mean(degree))
# print(stdev(degree))


from matplotlib import pyplot as plt

result_dir = "../result/"
epoch = "latest"
img_dir = os.path.join(result_dir, "final_plots", epoch)
os.makedirs(img_dir, exist_ok=True)

Geb_b30 = [11, 10, 12, 14, 16, 19, 17, 14, 18, 17]
years_b30 = range(2008,2018)
Geb_a30 = [12, 10, 13, 14, 12, 13, 18, 16]
years_a30 = range(2010,2018)

fig, ax = plt.subplots()
ax.plot(years_b30, Geb_b30, label='Prices 2008-2018', color='blue')
ax.plot(years_a30, Geb_a30, label='Prices 2010-2018', color = 'red')
legend = ax.legend(loc='center right', fontsize='x-large')
plt.xlabel('years')
plt.ylabel('prices')
plt.title('Comparison of the different prices')
plt.show()
plt.savefig(img_dir + "/" + 'temptest', bbox_inches='tight')
