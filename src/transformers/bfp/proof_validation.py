import numpy as np
import torch
from bfp_ops import _float_to_bfp, sparsity_unstructured, sparsity_hierarchial_n_m
from scipy.stats import wasserstein_distance

std = 1.0
M = 8
N = 8
n_samples = 1000
mant_bits = 5
epsilon = 1e-8
rounding_mode = "stoc"
sparsity_frac = 0.5
sparsity_mode = "unstr"

w_distances = np.zeros(n_samples)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

for i in range(n_samples):
    w = torch.randn(M, N, device=device, dtype=dtype) * std
    q_w =_float_to_bfp(w, mant_bits, epsilon, rounding_mode, device)
    if sparsity_mode == "unstr":
        s_w = sparsity_unstructured(w, device, sparsity_frac).reshape((M, N))
    else:
        s_w = sparsity_hierarchial_n_m(w, device, [2], [4]).reshape((M, N))
    q_tilde_w = torch.where(s_w == 0.0, w, q_w)
    q_w = q_w.flatten().detach().cpu().numpy().flatten()
    q_tilde_w = q_tilde_w.flatten().detach().cpu().numpy()
    w = w.flatten().detach().cpu().numpy()
    w_distances[i] = wasserstein_distance(q_w - w, q_tilde_w - w)

np.save(f"/parsadata1/lisa/experiments/proof_validation/err_{sparsity_mode}_hbfp{mant_bits + 1}.npy", w_distances)