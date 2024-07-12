import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest
import numpy as np
from .bfp_util import *
from .int_ops import Quantizer
from .mx_ops import *
# import mx
from .elemwise_ops import _quantize_fp

class rounding_modes:
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]

def round_tensor(t, mode, device):
    if mode == rounding_modes.STOC:
        sampled = torch.rand(t.shape, device=device) - 0.5
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exponent(t, epsilon):
    t = t.abs()
    max_v, _ = t.max(dim=1, keepdim=True)
    
    return (max_v + epsilon).log2().ceil()

def _convert_blocked_float_to_bfp(t, mant_bits, epsilon, rounding_mode, device):
    exp = get_exponent(t, epsilon)
    
    interval = torch.pow(2.0, exp-mant_bits)
    max_v = torch.pow(2.0, exp) - interval
    t = t/interval
    rounded = round_tensor(t, rounding_mode, device)
    rounded *= interval
    
    return torch.min(torch.max(rounded, -max_v), max_v)

def _convert_blocked_float_to_int(x, mant_bits, device):
    maxq = torch.tensor(2 ** mant_bits - 1).to(device)
    tmp = torch.zeros(x.shape[0], device=device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)
    xmax = torch.maximum(torch.abs(xmin), xmax)
    tmp = xmin < 0
    if torch.any(tmp):
        xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def _no_sparsity_float_to_bfp(t, block_size, mant_bits, epsilon, rounding_mode, device):
    orig_shape = t.shape
    padded_shape = list(orig_shape)
    
    if orig_shape[-1] % block_size != 0:
        pad_size = block_size - (orig_shape[-1] % block_size)
        t = F.pad(t, (0, pad_size), 'constant')
        padded_shape[-1] = orig_shape[-1] + pad_size
    
    t = t.contiguous().view(-1, block_size)
    t = _convert_blocked_float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    t = t.contiguous().view(padded_shape)
    
    return t.narrow(-1, 0, orig_shape[-1])

def _no_sparsity_float_to_int(t, block_size, mant_bits, device):
    orig_shape = t.shape
    padded_shape = list(orig_shape)
    
    if orig_shape[-1] % block_size != 0:
        pad_size = block_size - (orig_shape[-1] % block_size)
        t = F.pad(t, (0, pad_size), 'constant')
        padded_shape[-1] = orig_shape[-1] + pad_size
    
    t = t.contiguous().view(-1, block_size)
    quantizer = Quantizer()
    quantizer.configure(bits=mant_bits)
    quantizer.sym = False
    quantizer.find_params(t, weight=True)
    quant_t = quantizer.quantize(t)
    quant_t = quant_t.contiguous().view(padded_shape)
    
    return quant_t.narrow(-1, 0, orig_shape[-1])

def _unstructured_sparsity(t, device, sparsity_frac=0):
    assert (sparsity_frac > 0)
    
    orig_shape = t.shape
    temp = t.contiguous().view(1, -1)
    _, sparse_idx = torch.topk(torch.abs(temp), k=int(temp.shape[1]*sparsity_frac), dim=1, largest=False)
    zero_mask = torch.full(temp.shape, 1).to(device=device)
    zero_mask.scatter_(index=sparse_idx, dim=1, value=0)
    sparse_t = torch.where(zero_mask==0, 0, temp)
    
    return sparse_t.contiguous().view(orig_shape)

def _structured_N_M_sparsity(t, device, N=0, M=0):
    assert ((N > 0) and (M > 0) and (N <= M))

    orig_shape = t.shape
    padded_shape = list(orig_shape)

    if orig_shape[-1] % M != 0:
        pad_size = M - (orig_shape[-1] % M)
        t = F.pad(t, (0, pad_size), 'constant')
        padded_shape[-1] = orig_shape[-1] + pad_size

    temp = t.contiguous().view(-1, M)
    _, sparse_idx = torch.topk(torch.abs(temp), k=(M - N), dim=1, largest=False)
    zero_mask = torch.full(temp.shape, 1).to(device=device)
    zero_mask.scatter_(index=sparse_idx, dim=1, value=0)
    sparse_t = torch.where(zero_mask==0, 0, temp)
    sparse_t = sparse_t.contiguous().view(padded_shape)

    return sparse_t.narrow(-1, 0, orig_shape[-1])

def _sparsify(t, sparsity, sparsity_mode, device, N, M, sparsity_frac):
    if sparsity == True:
        if sparsity_mode == 'structured':
            return _structured_N_M_sparsity(t, device, N, M)
        elif sparsity_mode == 'unstructured':
            return _unstructured_sparsity(t, device, sparsity_frac)
        else:
            raise ValueError(f'Unknown sparsity mode: {sparsity_mode} given as argument')
    else:
        return t

def _quantize(t, num_format, block_size, mant_bits, epsilon, rounding_mode, device, identifier):
    if num_format == 'fp' and mant_bits == '32':
        return t
    elif num_format == 'bfp':
        return _no_sparsity_float_to_bfp(t, block_size, mant_bits, epsilon, rounding_mode, device)
    elif num_format == 'int':
        return _no_sparsity_float_to_int(t, block_size, mant_bits, device)
    else:
        raise ValueError(f'Unknown quantization format: {num_format}-{mant_bits} given as argument')

def float_to_bfp_blocked(t, a_num_format, a_mant_bits, w_num_format, w_mant_bits, epsilon, rounding_mode, device, block_size, 
                         in_sparsity, w_sparsity, grad_sparsity, sparsity_frac, N, M, sparsity_num_format, sparsify, first, sparsity_mode, 
                         identifier='', sgd_update=False, mx_w_elem_format='', mx_a_elem_format='', scale_bits=0, bfloat=0):

    # assert (num_format == 'bfp')
    assert (((sparsity_num_format == 'bfp') and (block_size > 0)) or (sparsity_num_format == 'fp') or (sparsity_num_format == 'int'))

    if identifier == 'in':
        sparsity_num_format = a_num_format
        mant_bits = a_mant_bits
    elif identifier == 'w':
        sparsity_num_format = w_num_format
        mant_bits = w_mant_bits

    if in_sparsity == True and identifier == 'in':
        sparsity = True
    elif w_sparsity == True and identifier == 'w':
        sparsity = True
    elif grad_sparsity == True and identifier == 'grad':
        sparsity = True
    else:
        sparsity = False
    
    # Enabled support of FP16/FP8 for activations
    if identifier == "in" and sparsity_num_format == "fp":
        if a_mant_bits == 16:
            q_t = t.to(torch.float16)
            q_t = q_t.to(torch.float32)
        elif a_mant_bits == 8:
            q_t = _quantize_fp(t, exp_bits=5, mantissa_bits=2)
        else: ValueError(f'FP{a_mant_bits} format for activations is not supported')
        return q_t
        
    if first == 's':
        sparse_t = _sparsify(t, sparsity, sparsity_mode, device, N, M, sparsity_frac)
        quant_t = _quantize(sparse_t, sparsity_num_format, block_size, mant_bits, epsilon, rounding_mode, device, identifier)
        return quant_t

    else:
        quant_t = _quantize(t, sparsity_num_format, block_size, mant_bits, epsilon, rounding_mode, device, identifier)
        sparse_t = _sparsify(quant_t, sparsity, sparsity_mode, device, N, M, sparsity_frac)
        return sparse_t

def MxM_pre_processing(x, w, transpose, **bfp_args):
    if transpose == True:
        return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), torch.transpose(float_to_bfp_blocked(torch.transpose(w, -1, -2), **bfp_args, identifier='w'), -1, -2))
    else:
        return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), float_to_bfp_blocked(w, **bfp_args, identifier='w'))

def _get_op_name(name, epsilon, w_mant_bits, rounding_mode, **kwargs):
    return  '%s_BFP_%s_%d' % (name, rounding_mode, w_mant_bits)

def _gen_bfp_op(op, name, bfp_args, transpose=False):
    name = _get_op_name(name, **bfp_args)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            return MxM_pre_processing(x, w, transpose, **bfp_args)

        @staticmethod
        def backward(ctx, grad_x, grad_w):
            return (grad_x, grad_w)

    NewOpIn.__name__ = name + '_In'
    new_op_in = NewOpIn.apply

    class NewOpOut(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op_out):
            return op_out

        @staticmethod
        def backward(ctx, op_out_grad):
            return float_to_bfp_blocked(op_out_grad, **bfp_args, identifier='grad')

    NewOpOut.__name__ = name + '_Out'
    new_op_out = NewOpOut.apply

    def new_op(x, w, *args, **kwargs):
        x, w = new_op_in(x, w)
        out = op(x, w, *args, **kwargs)
        return new_op_out(out)

    return new_op

def _get_bfp_op(op, name, bfp_args, transpose=False):
    _bfp_ops = {}
    op_name = _get_op_name(name, **bfp_args)
    if op_name not in _bfp_ops:
        _bfp_ops[name] = _gen_bfp_op(op, name, bfp_args, transpose)

    return _bfp_ops[name]

# def F_linear_bfp(**kwargs):
#     bfp_args = unpack_bfp_args(kwargs)
#     if bfp_args['num_format'] == 'bfp':
#         return _get_bfp_op(F.linear, 'linear', bfp_args)
#     else:
#         return F.linear

def F_matmul_bfp(**kwargs):
    self.sparsity_num_format = kwargs['num_format']
    self.sparsify = kwargs['sparsify']

    if self.sparsify == False and self.sparsity_num_format == 'fp':
        return torch.matmul
    elif self.sparsity_num_format in ['bfp', 'fp', 'int']:
        return _get_bfp_op(torch.matmul, 'matmul', kwargs, True)
    else:
        raise NotImplementedError('NumFormat not implemented')

class BFPConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.sparsity_num_format = kwargs['sparsity_num_format']
        self.sparsify = kwargs['sparsify']
        self.conv_op = _get_bfp_op(F.conv2d, 'Conv2d', kwargs)

    def forward(self, input):
        if self.sparsify == False and self.sparsity_num_format == 'fp':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.sparsity_num_format in ['bfp', 'fp', 'int']:
            conv = self.conv_op(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            return conv

        else:
            raise NotImplementedError('NumFormat not implemented')

class BFPLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        self.sparsity_num_format = kwargs['sparsity_num_format']
        print(list(kwargs.keys()))
        self.sparsify = kwargs['sparsify']
        self.linear_op = _get_bfp_op(F.linear, 'linear', kwargs)

    def forward(self, input):
        if self.sparsify == False and self.sparsity_num_format == 'fp':
            return F.linear(input, self.weight, self.bias)
    
        elif self.sparsity_num_format in ['bfp', 'fp', 'int']:
            l = self.linear_op(input, self.weight, self.bias)
            return l

        else:
            raise NotImplementedError('NumFormat not implemented')
