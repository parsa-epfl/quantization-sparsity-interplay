import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = "query"
N = 4
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
kqv_fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp_last_layer.pkl", "rb"))
kqv_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-sparse_last_layer.pkl", "rb"))
kqv_hbfp_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp8_last_layer.pkl", "rb"))
kqv_hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp8-sparse_last_layer.pkl", "rb"))

fp_tensor = kqv_fp_dict[mode][head_idx].flatten()
sparse_tensor = kqv_sparse_dict[mode][head_idx].flatten()
hbfp_tensor = kqv_hbfp_dict[mode][head_idx].flatten()
hbfp_sparse_tensor = kqv_hbfp_sparse_dict[mode][head_idx].flatten()


# fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig = plt.figure(figsize = (10, 8))
fig.suptitle("Kernel density estimation of " + mode + " projections: layer 11 head " + str(head_idx))

projections = [fp_tensor, sparse_tensor, hbfp_tensor, hbfp_sparse_tensor]
proj_names = ["fp32_dense", "fp32_sparse", "bfp8_dense", "bfp8_sparse"]

for i in range(N):
    data = projections[i]
    sns.kdeplot(data, bw_adjust=.5, label=proj_names[i])

plt.legend()
plt.savefig(PATH_TO_FOLDER + mode + 'kde_layer11_head' + str(head_idx) + '.png')    
