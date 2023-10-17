import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "query"
N = 4
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
kqv_fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp-last_layer.pkl", "rb"))
kqv_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-sparse-last_layer.pkl", "rb"))
kqv_hbfp_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-last_layer.pkl", "rb"))
kqv_hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-sparse-last_layer.pkl", "rb"))

fp_tensor = kqv_fp_dict[mode][head_idx].flatten()
sparse_tensor = kqv_sparse_dict[mode][head_idx].flatten()
hbfp_tensor = kqv_hbfp_dict[mode][head_idx].flatten()
hbfp_sparse_tensor = kqv_hbfp_sparse_dict[mode][head_idx].flatten()


# fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig = plt.figure(figsize = (10, 8))
ax = plt.gca()
fig.suptitle("Distributions of " + mode + " projections before wrapping: layer 11 head " + str(head_idx), fontsize="large")

projections = [fp_tensor, sparse_tensor, hbfp_tensor, hbfp_sparse_tensor]
proj_names = ["fp32_dense", "fp32_sparse", "bfp8_dense", "bfp8_sparse"]

plt.boxplot(projections, labels=proj_names, whis=2.8)
ax.set_xticks([y + 1 for y in range(len(proj_names))],
                labels=proj_names)
ax.set_ylabel("Observed_value")
ax.yaxis.grid(True)

PATH_TO_PIC = PATH_TO_FOLDER + "pics/"
plt.savefig(PATH_TO_PIC + mode + 'box_layer11_head' + str(head_idx) + '.png') 