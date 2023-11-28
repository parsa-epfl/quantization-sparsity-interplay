import os
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

PATH_TO_FOLDER = "/home/parsa_liza/experiments/layers/attention_maps_2/"
dense_scores_stats = np.zeros((12, 3))
dense_outputs_stats = np.zeros((12, 3))
dists_scores = np.zeros(12)
dists_outputs = np.zeros(12)
Nr = 1
Nc = 2

fig, axs = plt.subplots(Nr, Nc, figsize = (10, 8))
i = head_idx = 0
for dense_f, sparse_f in zip(os.listdir("/home/parsa_liza/experiments/layers/dense/"), os.listdir("/home/parsa_liza/experiments/layers/sparse/")):
    dense_dict = pickle.load(open("/home/parsa_liza/experiments/layers/dense/" + dense_f, "rb"))
    sparse_dict = pickle.load(open("/home/parsa_liza/experiments/layers/sparse/" + sparse_f, "rb"))

    dense_scores = dense_dict["scores_after_sfmax"][1]
    sparse_scores = sparse_dict["scores_after_sfmax"][1]

    for head_idx in range(12):
        images = []
        fig, axs = plt.subplots(Nr, Nc, figsize = (16, 10))
        fig.suptitle("Distributions of attention weights: layer " + str(i) + " head " + str(head_idx), fontsize="large")
        images.append(axs[0].imshow(dense_scores[head_idx][:128, :128]))
        images.append(axs[1].imshow(sparse_scores[head_idx][:128, :128]))
        axs[0].set_title("dense", fontsize="large")
        axs[1].set_title("sparse", fontsize="large")
        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
        plt.savefig(f"{PATH_TO_FOLDER}_{i}_{head_idx}")
        plt.close()
    print("Layer ", i, " processed!")
    i += 1
# for dense_f, sparse_f in zip(os.listdir("/home/parsa_liza/experiments/layers/dense/"), os.listdir("/home/parsa_liza/experiments/layers/sparse/")):
#     dense_dict = pickle.load(open("/home/parsa_liza/experiments/layers/dense/" + dense_f, "rb"))
#     sparse_dict = pickle.load(open("/home/parsa_liza/experiments/layers/sparse/" + sparse_f, "rb"))
#     print(dense_dict["scores_after_sfmax"].shape)
    
#     dense_scores = dense_dict["scores_after_sfmax"][:, 11].flatten()
#     sparse_scores = sparse_dict["scores_after_sfmax"][:, 11].flatten()
#     dists_scores[i] = wasserstein_distance(dense_scores, sparse_scores)

#     dense_o = dense_dict["attention_output"][:, 11].flatten()
#     sparse_o = sparse_dict["attention_output"][:, 11].flatten()
#     dists_outputs[i] = wasserstein_distance(dense_o, sparse_o)
#     i += 1

# for i, f_name in enumerate(os.listdir(PATH_TO_FOLDER)):
#     layer_dict = pickle.load(open(PATH_TO_FOLDER + f_name, "rb"))
#     scores = layer_dict["scores_after_sfmax"].flatten()
#     lower_q = np.quantile(scores, 0.25)
#     median = np.quantile(scores, 0.5)
#     higher_q = np.quantile(scores, 0.75)
#     dense_scores_stats[i] = [lower_q, median, higher_q]
    
#     outputs = layer_dict["attention_output"].flatten()
#     lower_q = np.quantile(outputs, 0.25)
#     median = np.quantile(outputs, 0.5)
#     higher_q = np.quantile(outputs, 0.75)
#     dense_outputs_stats[i] = [lower_q, median, higher_q]


# PATH_TO_FOLDER = "/home/parsa_liza/experiments/layers/sparse/"
# sparse_scores_stats = np.zeros((12, 3))
# sparse_outputs_stats = np.zeros((12, 3))

# for i, f_name in enumerate(os.listdir(PATH_TO_FOLDER)):
#     layer_dict = pickle.load(open(PATH_TO_FOLDER + f_name, "rb"))
#     scores = layer_dict["scores_after_sfmax"].flatten()
#     lower_q = np.quantile(scores, 0.25)
#     median = np.quantile(scores, 0.5)
#     higher_q = np.quantile(scores, 0.75)
#     sparse_scores_stats[i] = [lower_q, median, higher_q]

#     outputs = layer_dict["attention_output"].flatten()
#     lower_q = np.quantile(outputs, 0.25)
#     median = np.mean(outputs)
#     higher_q = np.quantile(outputs, 0.75)
#     sparse_outputs_stats[i] = [lower_q, median, higher_q]

# delta_scores = np.abs(dense_scores_stats - sparse_scores_stats)
# delta_outputs = np.abs(dense_outputs_stats - sparse_outputs_stats)
# print("Scores")
# for item in delta_scores:
#     print(item)

# print("Outputs")
# for item in delta_outputs:
#     print(item)

# PATH_TO_PIC_SCORES = "/home/parsa_liza/experiments/layers/scores_3.png"
# PATH_TO_PIC_OUTPUT = "/home/parsa_liza/experiments/layers/output_3.png"



# idx = np.arange(12)
# plt.plot(idx, dists_scores, linestyle='dashed')
# plt.scatter(idx, dists_scores)
# plt.xlabel("layer_idx")
# # plt.ylim((0.0, 1.6e-4))
# plt.title("Wass dist between dense and sparse attention weights")
# # plt.savefig(PATH_TO_PIC_SCORES)
