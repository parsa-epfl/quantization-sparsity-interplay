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
from .bfp_ops import fp32_sparsity_hierarchial_n_m, fp32_sparsity_unstructured

import sys
sys.path.append("/parsadata1/lisa/microxcaling/")
print(sys.path)

import mx
from mx import Linear, apply_mx_specs, finalize_mx_specs


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
                    print("I'm here")
                    self.weight = torch.nn.Parameter(fp32_sparsity_hierarchial_n_m(self.weight, self.device, self.N, self.M))
                else:
                    print("I'm hereee")
                    self.weight = torch.nn.Parameter(fp32_sparsity_unstructured(self.weight, self.device, self.sparsity_frac))
                self.sparsity_init = True
        return super().forward(inputs)
