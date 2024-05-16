import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import os
import pdb
import itertools as it
import logging
import unittest
import numpy as np
import pickle
from .bfp_ops import _unstructured_sparsity, _structured_N_M_sparsity

import mx
from mx import Linear
from mx import matmul as mx_matmul
from .specs import apply_mx_specs, finalize_mx_specs

class MXLinear(Linear):
    def __init__(self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        sparsity=False, 
        device=None,
        sparsity_mode="structured",
        sparsity_frac=0.0,
        N=[],
        M=[],
    ):
        mx_specs = apply_mx_specs(mx_specs)
        mx_specs = finalize_mx_specs(mx_specs)
        super().__init__(in_features, out_features, bias, mx_specs, name)
        assert(sparsity_mode in ["structured", "unstructured"])

        self.sparsity = sparsity
        self.device = device
        self.sparsity_mode = sparsity_mode
        self.sparsity_frac = sparsity_frac
        self.N = N 
        self.M = M
        self.sparsity_init = False

    def forward(self, inputs):
        if self.sparsity:
            if not self.sparsity_init:
                if self.sparsity_mode == "structured":
                    self.weight = torch.nn.Parameter(_structured_N_M_sparsity(self.weight, self.device, self.N[0], self.M[0]))
                else:
                    self.weight = torch.nn.Parameter(_unstructured_sparsity(self.weight, self.device, self.sparsity_frac))
                self.sparsity_init = True
        return super().forward(inputs)

def MXMatmul(in1, in2, mx_specs=None, sparsity=False, sparsity_mode="unstructured", device=None, N=[], M=[], sparsity_frac=0):
    assert(sparsity_mode in ["structured", "unstructured"])
    if sparsity:
        if sparsity_mode == "structured":
            in2 = torch.transpose(_structured_N_M_sparsity(torch.transpose(in2, -1, -2), device, N[0], M[0]), -1, -2)
        else:
            in2 = torch.transpose(_unstructured_sparsity(torch.transpose(in2, -1, -2), device, sparsity_frac), -1, -2)
    return mx_matmul(in1, in2, mx_specs=mx_specs)
        

