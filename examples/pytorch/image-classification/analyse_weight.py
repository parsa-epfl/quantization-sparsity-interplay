import torch
import numpy as np
import sys
from scipy import fftpack
import matplotlib.pyplot as plt

path = "./sparse_results/cifar10/baseline/cifar10_fp32/pytorch_model.bin"

model = torch.load(path)
for layer, weight in model.items():
    if len(list(weight.shape)) == 2:
        weight_mat = weight.cpu().detach().numpy()
