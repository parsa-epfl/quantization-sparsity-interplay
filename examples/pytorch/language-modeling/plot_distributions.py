import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "attention_output"
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp-dense-attention-outputs-12-layer.pkl", "rb"))
hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-1:8-hbfp8-attention-outputs-12-layer.pkl", "rb"))

fp_tensor = fp_dict[mode][head_idx]
hbfp_sparse_tensor = hbfp_sparse_dict[mode][head_idx]
Nr = 1
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig.suptitle("Distributions of attention output: layer 11 head " + str(head_idx), fontsize="large")

images = []
xmin, xmax = -2, 2
ymin, ymax = 0, 600
projections = [fp_tensor, hbfp_sparse_tensor]
proj_names = ["fp32_dense", "bfp8_sparse1:8"]
for j in range(Nc):
    data = projections[j]
    images.append(axs[j].hist(data))
    axs[j].set_title(proj_names[j], fontsize="medium")
    axs[j].set_xlim([xmin, xmax])
    axs[j].set_ylim([ymin, ymax])
    #axs[i, j].label_outer()
PATH_TO_PIC = PATH_TO_FOLDER + "pics/"
plt.savefig(PATH_TO_PIC + mode + '1:8-hbfp8-distr_layer11.png')
