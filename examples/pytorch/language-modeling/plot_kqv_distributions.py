import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
mode = ""
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
# kqv_fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp-last_layer.pkl", "rb"))
# kqv_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-sparse-last_layer.pkl", "rb"))
# kqv_hbfp_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-last_layer.pkl", "rb"))
# kqv_hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-sparse-last_layer.pkl", "rb"))
kq_before_wrapping_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp8-sparse.pkl", "rb"))
kq_after_wrapping_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-sparse-kq-wrapped.pkl", "rb"))


fp_k = kq_before_wrapping_dict["key"][head_idx].flatten()
fp_q = kq_before_wrapping_dict["query"][head_idx].flatten()
# sparse_tensor = kqv_sparse_dict[mode][head_idx]
# hbfp_tensor = kqv_hbfp_dict[mode][head_idx]
bfp_q = kq_after_wrapping_dict["query_wrapped"][head_idx].flatten()
bfp_sparse_k = kq_after_wrapping_dict["key_wrapped"][head_idx].flatten()

bfp_q = bfp_q.detach().cpu().numpy()
bfp_sparse_k = bfp_sparse_k.detach().cpu().numpy()


Nr = 2
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
fig.suptitle("Distributions of projections before and after wrapping: layer 0 head " + str(head_idx), fontsize="large")

images = []
xmin, xmax = -4.0, 4.0
ymin, ymax = 0, 2000
projections = [fp_k, fp_q, bfp_sparse_k, bfp_q]
proj_names = ["key_fp", "query_fp", "key_bfp_sparse", "query_bfp"]
for i in range(Nr):
    for j in range(Nc):
        data = projections[2*i+j]
        images.append(axs[i, j].hist(data, bins=100))
        axs[i, j].set_title(proj_names[2*i+j], fontsize="medium")
        axs[i, j].set_xlim([xmin, xmax])
        axs[i, j].set_ylim([ymin, ymax])
        #axs[i, j].label_outer()
PATH_TO_PIC = PATH_TO_FOLDER + "pics/"
plt.savefig(PATH_TO_PIC + mode + 'distr_layer0_wrapped' + str(head_idx) + '.png')
