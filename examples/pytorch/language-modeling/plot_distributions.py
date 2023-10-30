import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "scores_after_sfmax"
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp-dense-attention-outputs.pkl", "rb"))
hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-sparse-attention-outputs.pkl", "rb"))

fp_tensor = fp_dict[mode][head_idx][0:6].flatten()
hbfp_sparse_tensor = hbfp_sparse_dict[mode][head_idx][0:6].flatten()
print(fp_tensor.shape)
print(hbfp_sparse_tensor.shape)
Nr = 1
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig.suptitle("Distributions of attention scores: layer 0 head " + str(head_idx), fontsize="large")

images = []
xmin, xmax = 0.0, 0.014
ymin, ymax = 0, 630
projections = [fp_tensor, hbfp_sparse_tensor]
custom_bins = [100, 80]
proj_names = ["fp32_dense", "fp32_sparse"]
for j in range(Nc):
    data = projections[j]
    n, bins, patches = axs[j].hist(data, bins=custom_bins[j], color=['b'])
    print(n)
    print(sum(n))
    #axs[j].set_title(proj_names[j], fontsize="medium")
    axs[j].set_xlim([xmin, xmax])
    axs[j].set_ylim([ymin, ymax])
    #axs[i, j].label_outer()
PATH_TO_PIC = PATH_TO_FOLDER + "pics/"
plt.savefig(PATH_TO_PIC + mode + 'scores_fp_blue.png')
