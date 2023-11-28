import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "attention_output"
PATH_TO_FOLDER = "/home/parsa_liza/experiments/layers/"
fp_dict = pickle.load(open(PATH_TO_FOLDER + "dense/2.pkl", "rb"))
sparse_dict = pickle.load(open(PATH_TO_FOLDER + "sparse/2.pkl", "rb"))

fp_tensor = fp_dict[mode].flatten()
sparse_tensor = sparse_dict[mode].flatten()
print(fp_tensor.shape)
print(sparse_tensor.shape)
Nr = 1
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig.suptitle("Distributions of attention outputs: layer 11 head " + str(head_idx), fontsize="large")

images = []
# xmin, xmax = -2.0, 2.0
# ymin, ymax = 0, 3000
projections = [fp_tensor, sparse_tensor]
# custom_bins = [5000, 1000]
proj_names = ["fp32_dense", "fp32_sparse"]
# for j in range(Nc):
    # data = projections[j]
    # n, bins, patches = axs[j].hist(data, bins=custom_bins[j], color=['b'])
    # print(n)
    # print(sum(n))
    # axs[j].set_title(proj_names[j], fontsize="medium")
    # axs[j].set_xlim([xmin, xmax])
    # axs[j].set_ylim([ymin, ymax])
    #axs[i, j].label_outer()
fig = plt.figure(figsize = (10, 8))
ax = plt.gca()
fig.suptitle("Distributions of attention outputs before wrapping: layer 0 head 11", fontsize="large")
props = {"alpha": 0.5}
plt.boxplot(projections, labels=proj_names, whis=(1, 99), flierprops=props)
ax.set_xticks([y + 1 for y in range(len(proj_names))],
                labels=proj_names)
ax.set_ylabel("Observed_value")
ax.yaxis.grid(True)
PATH_TO_PIC = "/home/parsa_liza/experiments/layers/box_outputs_4.png"
plt.savefig(PATH_TO_PIC)
