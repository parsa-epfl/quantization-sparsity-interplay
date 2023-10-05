import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "query"
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
kqv_fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp.pkl", "rb"))
kqv_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-sparse.pkl", "rb"))
kqv_hbfp_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp8.pkl", "rb"))
kqv_hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp8-sparse.pkl", "rb"))

fp_tensor = kqv_fp_dict[mode][head_idx].flatten()
sparse_tensor = kqv_sparse_dict[mode][head_idx].flatten()
hbfp_tensor = kqv_hbfp_dict[mode][head_idx].flatten()
hbfp_sparse_tensor = kqv_hbfp_sparse_dict[mode][head_idx].flatten()

Nr = 2
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig.suptitle("Kernel density estimation of " + mode + " projections: layer 0 head " + str(head_idx), fontsize="large")

images = []
projections = [fp_tensor, sparse_tensor, hbfp_tensor, hbfp_sparse_tensor]
proj_names = ["fp32_dense", "bfp8_dense", "fp32_sparse", "bfp8_sparse"]

for i in range(Nr):
    for j in range(Nc):
        data = projections[2*i+j]
        sns.kdeplot(data, ax=axs[i, j], bw_adjust=.5)
        axs[i, j].set_title(proj_names[2*i+j], fontsize="medium")
        #axs[i, j].label_outer()
plt.savefig(PATH_TO_FOLDER + mode + 'kde_layer0_head' + str(head_idx) + '.png')