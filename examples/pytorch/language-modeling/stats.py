import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


PATH_TO_FOLDER = "/home/parsa_liza/experiments/layers/dense/"
dense_scores_stats = np.zeros((12, 3))
dense_outputs_stats = np.zeros((12, 3))
for i, f_name in enumerate(os.listdir(PATH_TO_FOLDER)):
    layer_dict = pickle.load(open(PATH_TO_FOLDER + f_name, "rb"))
    scores = layer_dict["scores_after_sfmax"].flatten()
    lower_q = np.quantile(scores, 0.25)
    median = np.quantile(scores, 0.5)
    higher_q = np.quantile(scores, 0.75)
    dense_scores_stats[i] = [lower_q, median, higher_q]

    outputs = layer_dict["attention_output"].flatten()
    lower_q = np.quantile(outputs, 0.25)
    median = np.quantile(outputs, 0.5)
    higher_q = np.quantile(outputs, 0.75)
    dense_outputs_stats[i] = [lower_q, median, higher_q]


PATH_TO_FOLDER = "/home/parsa_liza/experiments/layers/sparse/"
sparse_scores_stats = np.zeros((12, 3))
sparse_outputs_stats = np.zeros((12, 3))

for i, f_name in enumerate(os.listdir(PATH_TO_FOLDER)):
    layer_dict = pickle.load(open(PATH_TO_FOLDER + f_name, "rb"))
    scores = layer_dict["scores_after_sfmax"].flatten()
    lower_q = np.quantile(scores, 0.25)
    median = np.quantile(scores, 0.5)
    higher_q = np.quantile(scores, 0.75)
    sparse_scores_stats[i] = [lower_q, median, higher_q]

    outputs = layer_dict["attention_output"].flatten()
    lower_q = np.quantile(outputs, 0.25)
    median = np.mean(outputs)
    higher_q = np.quantile(outputs, 0.75)
    sparse_outputs_stats[i] = [lower_q, median, higher_q]

delta_scores = dense_scores_stats - sparse_scores_stats
delta_outputs = np.abs(dense_outputs_stats - sparse_outputs_stats)
print("Scores")
for item in delta_scores:
    print(item)

print("Outputs")
for item in delta_outputs:
    print(item)

PATH_TO_PIC_SCORES = "/home/parsa_liza/experiments/layers/scores.png"
PATH_TO_PIC_OUTPUT = "/home/parsa_liza/experiments/layers/output.png"

idx = np.arange(12)
plt.plot(idx, delta_outputs[:, 0], linestyle='dashed')
plt.plot(idx, delta_outputs[:, 1], linestyle='dashed')
plt.plot(idx, delta_outputs[:, 2], linestyle='dashed')
plt.scatter(idx, delta_outputs[:, 0], label="1st_quartile")
plt.scatter(idx, delta_outputs[:, 1], label="median")
plt.scatter(idx, delta_outputs[:, 2], label="3rd_quartile")
plt.legend()
plt.xlabel("layer_idx")
plt.ylabel("output_dense - output_sparse|")
plt.title("Absolute difference between dense and sparse attention outputs")
plt.savefig(PATH_TO_PIC_OUTPUT) 