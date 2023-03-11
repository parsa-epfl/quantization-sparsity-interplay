# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL &
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest

class rounding_modes:
    """
    When converting fp32 tensors to bfp, the rounding mode can be chosen.
    STOC: Stochastic rounding
    DETERM: Deterministic rounding
    """
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]

def round_tensor(t, mode, device):
    """
    Perform the rounding of the tensor t by using selected mode
    """
    if mode == rounding_modes.STOC:
        if device == "cpu":
            sampled = torch.FloatTensor(t.size(), device = device).uniform_(-0.5, 0.5)
        else:
            sampled = torch.cuda.FloatTensor(t.size()).uniform_(-0.5, 0.5)
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exponent(t, epsilon):
    """
    Find the shared exponent of the tensor t.
    The exponent of the largest tensor value is selected as the shared exponent.
    """
    #Exponent is independent of the sign
    t = t.abs()
    #Find the maximum element of the tensor t
    max_v, _ = t.max(dim=1, keepdim=True)
    #Get the exponent of that element (We use ceil because in bfp format, we convert using 0.mantissa_bits instead of fp32's 1.mantissa_bits)
    return (max_v + epsilon).log2().ceil()

def _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device, sparsity=False, sparsity_frac=0, N=0, M=0, cols=0, exp_given=None):
    """
    Convert float tensor t to bfp
    """
    #print(f'mant: {mant_bits}')
    #print(t.shape)
    #print(t)

    # # No sparsity at all
    # exp = get_exponent(t, epsilon)
    # #The interval between two consecutive numbers with that exponent value
    # interval = torch.pow(2.0, exp-mant_bits)
    # #The maximum representable value with exp
    # max_v = torch.pow(2.0, exp) - interval

    # # To ensure that we preserve the interval
    # t = t/interval
    # rounded = round_tensor(t, rounding_mode, device)
    # rounded *=  interval

    # #To ensure that there is no underflow or overflow
    # new_t = torch.min(torch.max(rounded, -max_v), max_v).to(device=device)
    # return new_t

    # # Sparsity scheme 1: Zero out the k% of the blocks with minimal exponents
    # exp = get_exponent(t, epsilon)
    # _, sparse_idx = torch.topk(exp, k=int(exp.shape[0]*sparsity_frac), largest=False, dim=0)
    # zero_mask = torch.full(exp.shape, 1).to(device=device)
    # if sparsity == True:
    #     zero_mask.scatter_(index=sparse_idx, dim=0, value=0)

    # #The interval between two consecutive numbers with that exponent value
    # interval = torch.pow(2.0, exp-mant_bits)
    # #The maximum representable value with exp
    # max_v = torch.pow(2.0, exp) - interval

    # # To ensure that we preserve the interval
    # t = t/interval
    # rounded = round_tensor(t, rounding_mode, device)
    # rounded *=  interval

    # #To ensure that there is no underflow or overflow
    # new_t = torch.min(torch.max(rounded, -max_v), max_v).to(device=device)
    # return torch.where(zero_mask==0, 0, new_t)

    # # Sparsity scheme 2: In each row, zero out the block with lowest exponent
    # exp = get_exponent(t, epsilon)
    # if sparsity == True:
    #     if cols == 0:
    #         raise NotImplementedError("Invalid no. of coloumns passed")
    #     else:
    #         block_size = t.shape[-1]
    #         row_exps = exp.view(-1, cols//block_size)
    #         row_exps[row_exps == row_exps.min(dim=1, keepdims=True).values] = 0
    #         zero_mask = row_exps.view(-1, 1)
    # else:
    #     zero_mask = torch.full(exp.shape, 1).to(device=device)

    # #The interval between two consecutive numbers with that exponent value
    # interval = torch.pow(2.0, exp-mant_bits)
    # #The maximum representable value with exp
    # max_v = torch.pow(2.0, exp) - interval

    # # To ensure that we preserve the interval
    # t = t/interval
    # rounded = round_tensor(t, rounding_mode, device)
    # rounded *=  interval

    # #To ensure that there is no underflow or overflow
    # new_t = torch.min(torch.max(rounded, -max_v), max_v).to(device=device)
    # return torch.where(zero_mask==0, 0, new_t)

    # # Sparsity Scheme 3: Unstructured sparsity, sparsify out F% of all quantized elements inside a tensor
    # exp = get_exponent(t, epsilon)
    # #The interval between two consecutive numbers with that exponent value
    # interval = torch.pow(2.0, exp-mant_bits)
    # #The maximum representable value with exp
    # max_v = torch.pow(2.0, exp) - interval

    # # To ensure that we preserve the interval
    # t = t/interval
    # rounded = round_tensor(t, rounding_mode, device)
    # rounded *=  interval

    # #To ensure that there is no underflow or overflow
    # t = torch.min(torch.max(rounded, -max_v), max_v).to(device=device)

    # temp = t.contiguous().view(-1, 1)
    # _, sparse_idx = torch.topk(torch.abs(temp), k=int(temp.shape[0]*sparsity_frac), dim=0, largest=False)
    # zero_mask = torch.full(temp.shape, 1).to(device=device)

    # if sparsity == True:
    #     zero_mask.scatter_(index=sparse_idx, dim=0, value=0)

    # return torch.where(zero_mask==0, 0, temp)

    # Sparsity Scheme 4: N:M sparsity inside each block
    block_size = t.shape[1]
    assert (N < M) and (N!=0) and (M!=0) and (M<=block_size)
    exp = get_exponent(t, epsilon)
    #The interval between two consecutive numbers with that exponent value
    interval = torch.pow(2.0, exp-mant_bits)
    #The maximum representable value with exp
    max_v = torch.pow(2.0, exp) - interval

    # To ensure that we preserve the interval
    t = t/interval
    rounded = round_tensor(t, rounding_mode, device)
    rounded *=  interval

    #To ensure that there is no underflow or overflow
    t = torch.min(torch.max(rounded, -max_v), max_v).to(device=device)

    if M != block_size:
        t = torch.reshape(t, (1, -1))
        pad_size = M - (t.shape[1] % M)
        t = F.pad(t, (0, pad_size), 'constant')
        t = torch.reshape(t, (-1, M))

    temp_t = torch.abs(t)
    _, sparse_idx = torch.topk(temp_t, k=N, dim=1, largest=False)
    zero_mask = torch.full(temp_t.shape, 1).to(device=device)

    if sparsity == True:
        zero_mask.scatter_(index=sparse_idx, dim=1, value=0)

    t = torch.where(zero_mask==0, 0, t)
    
    if M != block_size:
        t = torch.reshape(t, (1, -1))
        t = t.narrow(-1, 0, (t.shape[1]-pad_size))
    
    return t

def float_to_bfp_blocked(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25, bfp_block_size=0,
                       num_format='', weight_mant_bits=0, in_sparsity=False, w_sparsity=False, grad_sparsity=False, rearrange=False, 
                       sparsity_frac=0, N=0, M=0, sparsity_num_format='bfp', identifier='',
                       sgd_update=False, mant_bits_pow=None):
    """
    Convert fp32 tensor t to bfp with blocks.
    Used for weights (which are handled in the optimizer)
    """

    assert num_format == 'bfp'

    if in_sparsity == True and identifier == 'in':
        sparsity = True
    elif w_sparsity == True and identifier == 'w':
        sparsity = True
    elif grad_sparsity == True and identifier == 'grad':
        sparsity = True
    else:
        sparsity = False

    assert ((sparsity==False) or ((sparsity==True)and(sparsity_frac!=0)))

    if sparsity_num_format == 'fp32':
        # # Scheme 1: Remove F% of all elements
        # temp = t.contiguous().view(1, -1).float()
        # _, sparse_idx = torch.topk(torch.abs(temp), k=int(temp.shape[1]*sparsity_frac), dim=1, largest=False)
        # zero_mask = torch.full(temp.shape, 1).to(device=device)

        # if sparsity == True:
        #     zero_mask.scatter_(index=sparse_idx, dim=1, value=0)

        # temp = torch.where(zero_mask==0, 0, temp)
        # return temp.contiguous().view(t.shape)

        # Scheme 2: N:M sparsity
        orig_shape = t.shape
        t = torch.reshape(t, (1, -1))
        pad_size = M - (t.shape[1] % M)
        t = F.pad(t, (0, pad_size), 'constant')
        t = torch.reshape(t, (-1, M))

        temp_t = torch.abs(t)
        _, sparse_idx = torch.topk(temp_t, k=N, dim=1, largest=False)
        zero_mask = torch.full(temp_t.shape, 1).to(device=device)

        if sparsity == True:
            zero_mask.scatter_(index=sparse_idx, dim=1, value=0)

        t = torch.where(zero_mask==0, 0, t)
        
        t = torch.reshape(t, (1, -1))
        t = t.narrow(-1, 0, (t.shape[1]-pad_size))
        return torch.reshape(t, orig_shape)

    elif sparsity_num_format == 'bfp':
        if sgd_update:
            mant_bits = weight_mant_bits

        orig_shape = t.shape
        block_size = bfp_block_size
        if block_size == 0:
            return _float_to_bfp(t.view(1, -1), mant_bits, epsilon, rounding_mode, device, sparsity, sparsity_frac, N=N, M=M).view(orig_shape)

        padded_shape = list(orig_shape)

        if orig_shape[-1] % block_size != 0:
            pad_size = block_size - (orig_shape[-1] % block_size)
            t = F.pad(t, (0,pad_size),'constant')
            padded_shape[-1] = orig_shape[-1]+pad_size

        t = t.contiguous().view(-1,block_size)
        t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device, sparsity, sparsity_frac, N=N, M=M)
        t = t.contiguous().view(padded_shape)
        
        return t.narrow(-1,0,orig_shape[-1])
    else:
        raise NotImplementedError('NumFormat not implemented')

def calc_score(mat, device):
    new_mat = torch.abs(mat)
    n_cols = mat.shape[-1]
    score_mat = torch.zeros(mat.shape[0], n_cols).to(device)
    for idx in range(0, n_cols):
        col1 = new_mat[:, :, idx]
        for jdx in range(idx+1, n_cols):
            col2 = new_mat[:, :, jdx]
            comp1 = torch.le(col1, col2)
            comp2 = torch.le(col2, col1)
            score_mat[:, idx] += torch.sum(comp1, dim = -1)
            score_mat[:, jdx] += torch.sum(comp2, dim = -1)
    return score_mat

def rearrange_mats(matA, matB, device):
    orig_shape_A, orig_shape_B = matA.shape, matB.shape
    print(orig_shape_A, orig_shape_B)
    matA = matA.reshape(-1, matA.shape[-2], matA.shape[-1])
    matB = matB.reshape(-1, matB.shape[-2], matB.shape[-1])
    
    score_mat = calc_score(matB, device)
    max_idx = torch.argmax(score_mat, dim = -1)
    sort_idx = torch.abs(matB[torch.arange(matB.shape[0]), :, max_idx]).argsort(dim=-1)
    matB = matB[torch.arange(matB.shape[0])[:, None], sort_idx]
    matA = (torch.transpose(matA, -1, -2)[torch.arange(matA.shape[0])[:, None], sort_idx]).transpose(-1, -2)
    
    matA = matA.reshape(orig_shape_A)
    matB = matB.reshape(orig_shape_B)
    return matA, matB

def MxM_pre_processing(x, w, transpose, **bfp_args):
    device = bfp_args['device']
    rearrange = bfp_args['rearrange']
    if transpose == True:
        if rearrange == True:
            new_x, new_w = rearrange_mats(x, w, device)
            return (float_to_bfp_blocked(new_x, **bfp_args, identifier='in'), torch.transpose(float_to_bfp_blocked(torch.transpose(new_w, -1, -2), **bfp_args, identifier='w'), -1, -2))
        else:
            return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), torch.transpose(float_to_bfp_blocked(torch.transpose(w, -1, -2), **bfp_args, identifier='w'), -1, -2))
    else:
        if rearrange == True:
            new_x, new_w = rearrange_mats(x, torch.transpose(w, -1, -2), device)
            return (float_to_bfp_blocked(new_x, **bfp_args, identifier='in'), float_to_bfp_blocked(torch.transpose(new_w, -1, -2), **bfp_args, identifier='w'))
        else:
            return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), float_to_bfp_blocked(w, **bfp_args, identifier='w'))

def float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                         num_format='', weight_mant_bits=''):
    """
    Convert a batch of fp32 tensor t to bfp
    """
    """
    assert num_format == 'bfp'
    orig_shape = t.size()

    t = t.reshape(t.size()[0], -1)
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)
    """
    assert num_format == 'bfp'
    orig_shape = t.size()
    print(orig_shape)

    t = t.reshape(-1,orig_shape[-1])
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)

def float_to_bfp_batched_weight(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                         num_format='', weight_mant_bits=''):
    """
    Convert a batch of fp32 tensor t to bfp
    """
    assert num_format == 'bfp'
    orig_shape = t.size()

    # t = t.view(t.size()[0], -1)
    #print('------------------- in weight batched -------------------')
    #print(t.shape)
    #print(t)

    t = t.reshape(t.size()[0], -1)
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)


def tensor_to_tiled(t, orig_shape, bfp_tile_size):
    """
    Handle the tiling process.
    Output: the tiled tensor, the number of tiles in each dimension, the dimensions before and after the tiling to help 'untiling'
    """
    t = t.view(orig_shape[0], -1)
    matrix_h, matrix_w = t.size()

    numberOf_h_tiles = (matrix_h + bfp_tile_size - 1) // bfp_tile_size
    numberOf_w_tiles = (matrix_w + bfp_tile_size - 1) // bfp_tile_size

    matrix_h_pad = numberOf_h_tiles*bfp_tile_size
    matrix_w_pad = numberOf_w_tiles*bfp_tile_size

    h_pad = matrix_h_pad - matrix_h
    w_pad = matrix_w_pad - matrix_w

    t = F.pad(t, (0, w_pad, 0, h_pad),'constant')
    # t <-numberOf_h_tiles, tile_h, matrix_w
    t = t.view(numberOf_h_tiles, bfp_tile_size, matrix_w_pad)
    # t <- numberOf_h_tiles, matrix_w, tile_h,
    t.transpose_(1, 2)
    return (t.contiguous().view(numberOf_h_tiles*numberOf_w_tiles, -1),
            numberOf_h_tiles, numberOf_w_tiles,
            matrix_h, matrix_w,
            matrix_h_pad, matrix_w_pad)

def tiled_to_tensor(t, orig_shape, bfp_tile_size,
                    numberOf_h_tiles, numberOf_w_tiles,
                    matrix_h, matrix_w,
                    matrix_h_pad, matrix_w_pad):
    """
    Turn the tensor back to its shape before tiling
    """
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_w, tile_h
    t = t.view(numberOf_h_tiles, numberOf_w_tiles, bfp_tile_size, bfp_tile_size)
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_h, tile_w
    t.transpose_(2, 3)
    # t <- numberOf_h_tiles, tile_h, numberOf_w_tiles, tile_w
    t.transpose_(1, 2)
    t = t.contiguous().view(matrix_h_pad, matrix_w_pad)
    return t.narrow(0, 0, matrix_h).narrow(1, 0, matrix_w).view(orig_shape)


def float_to_bfp_tiled(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                       num_format='', weight_mant_bits=0,
                       sgd_update=False, mant_bits_pow=None):
    """
    Convert fp32 tensor t to bfp with tiling.
    Used for weights (which are handled in the optimizer)
    """
    assert num_format == 'bfp'
    if sgd_update:
        mant_bits = weight_mant_bits

    orig_shape = t.size()
    print(orig_shape)
    if bfp_tile_size == 0:
        return _float_to_bfp(t.view(1, -1), mant_bits, epsilon, rounding_mode, device).view(orig_shape)

    (t, numberOf_h_tiles, numberOf_w_tiles, matrix_h, matrix_w,
        matrix_h_pad, matrix_w_pad) = tensor_to_tiled(t, orig_shape, bfp_tile_size)

    t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)

    return tiled_to_tensor(t, orig_shape, bfp_tile_size,
                           numberOf_h_tiles, numberOf_w_tiles,
                           matrix_h, matrix_w,
                           matrix_h_pad, matrix_w_pad)

def _get_op_name(name, epsilon, mant_bits, rounding_mode, **kwargs):
    """
    Returns the operation's name that is performed in BFP format
    """
    return  '%s_BFP_%s_%d' % (name, rounding_mode, mant_bits)

def _gen_bfp_op(op, name, bfp_args, transpose=False):
    """
    Do the 'sandwich'
    With an original op:
    out = op(x, y)
    grad_x, grad_y = op_grad(grad_out)
    To the following:
    x_, y_ = input_op(x, y)
    Where input_op(x, y) -> bfp(x), bfp(y)
    and input_op_grad(grad_x, grad_y) -> bfp(grad_x), bfp(grad_y)
    out_ = op(x_, y_)
    out = output_op(out)
    Where output_op(out) -> bfp(out)
    and output_op_grad(grad_out) -> bfp(grad_out)
    This way we garantee that everything in and out of the forward and backward operations is
    properly converted to bfp
    """

    name = _get_op_name(name, **bfp_args)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            # return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), float_to_bfp_blocked(w, **bfp_args, identifier='w'))
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


_bfp_ops = {}


def _get_bfp_op(op, name, bfp_args, transpose=False):
    """
    Create the bfp version of the operation op
    This function is called when a bfp layer is defined. See BFPConv2d and BFPLinear below
    """
    op_name = _get_op_name(name, **bfp_args)
    if op_name not in _bfp_ops:
        _bfp_ops[name] = _gen_bfp_op(op, name, bfp_args, transpose)

    return _bfp_ops[name]


def unpack_bfp_args(kwargs):
    """
    Set up the bfp arguments
    """
    bfp_args = {}
    bfp_argn = [('num_format', 'fp32'),
                ('sparsity_num_format', 'fp32'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('mant_bits', 0),
                ('bfp_tile_size', 0),
                ('bfp_block_size', 0),
                ('weight_mant_bits', 0),
                ('in_sparsity', False),
                ('w_sparsity', False),
                ('grad_sparsity', False),
                ('N', 0),
                ('M', 0),
                ('rearrange', False),
                ('sparsity_frac', 0),
                ('device', 'cpu')]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args


def F_linear_bfp(**kwargs):
    """
    bfp linear function
    To be used in the model where F.linear is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        return _get_bfp_op(F.linear, 'linear', bfp_args)
    else:
        return F.linear

### TODO: Check the groupings
def F_matmul_bfp(**kwargs):
    """
    bfp matmul function
    To be used in the model where torch.matmul is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        # print("************************************* BFP MATMUL *****************************************")
        return _get_bfp_op(torch.matmul, 'matmul', bfp_args, True)
    else:
        return torch.matmul



class BFPConv2d(torch.nn.Conv2d):
    """
    bfp convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.num_format = self.bfp_args['num_format']
        self.conv_op = _get_bfp_op(F.conv2d, 'Conv2d', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.num_format == 'bfp':
            conv = self.conv_op(input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)
            if self.bias is not None:
                #print(f'shape conv: {conv.shape}')
                #print(f'bias conv: {self.bias.shape}')
                return conv + self.bias
            else:
                return conv

        else:
            raise NotImplementedError('NumFormat not implemented')


class BFPLinear(torch.nn.Linear):
    """
    bfp linear layer
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        self.linear_op = _get_bfp_op(F.linear, 'linear', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfp':
            l = self.linear_op(input, self.weight, None)
            if self.bias is not None:
                return l + self.bias
            else:
                return l

        else:
            raise NotImplementedError('NumFormat not implemented')


class TestCases(unittest.TestCase):
    def setUp(self):
        """
        Generate all possible bfp numbers that can be represented with given mantissa bits
        Note that we generate the bfp numbers using 0.mantissa_bits instead of fp32's 1.mantissa_bits)
        The implementation of HBFPRepresentables class and representable_numbers function has been adapted from
        https://github.com/TuringMachinegun/float_visualizer/blob/master/visualizer.py
        """
        class HBFPRepresentables():
            def __init__(self, sign, mantissa, exponent):
                self.sign = -1 if sign == "-" else 1
                self.exponent = exponent
                self.bias = 2**(len(exponent)-1)
                self.mantissa = "0" + mantissa

                self.exp_bits = len(exponent)
                self.mant_bits = len(mantissa)

            def to_float(self):
                mantissa_float = self.sign * int(self.mantissa,2)
                mantissa_float /= float(2**self.mant_bits)
                exponent_float = 2**(int(self.exponent, 2)-self.bias)
                return mantissa_float * exponent_float


        def representable_numbers(mant_bits, exp_bits = 10):
            possible_signs = ["-", "+"]
            possible_exponents = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=exp_bits)]
            possible_hbfp_mantissas = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=mant_bits)]
            bfp_representable_numbers = []

            for sign in possible_signs:
                for exponent in possible_exponents:
                    numbers_list = []
                    for mantissa in possible_hbfp_mantissas:
                        number = HBFPRepresentables(sign, mantissa, exponent)
                        numbers_list.append(number.to_float())

                    bfp_representable_numbers.append(numbers_list)

            bfp_representable_numbers = np.array(bfp_representable_numbers)
            return bfp_representable_numbers
        self.bfp = representable_numbers

    def test_float_to_bfp(self):
        """
        Generate random fp32 tensors
        Convert them to bfp
        Check if the converted values are contained in the possible bfp numbers
        """
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = 0
        rounding_mode = 'determ'

        for mant_bits in range(12):
            mant_bits +=1
            bfp_numbers = self.bfp(mant_bits)
            for i in range(10):
                t = torch.randn(10, 10, device=device, dtype=dtype)
                b=_float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                #print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))

    def test_tiled_and_batched(self):
        """
        Generate random fp32 tensors
        Convert them to bfp by using tiled and batched functions
        Check if the converted values are contained in the possible bfp numbers
        """
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = 0
        rounding_mode = 'determ'
        num_format='bfp'
        matrix_h, matrix_w = 32, 32
        tile_size = 15

        for mant_bits in range(12):
            mant_bits +=1
            bfp_numbers = self.bfp(mant_bits)
            for i in range(10):
                t = torch.randn(matrix_h, matrix_w, device=device, dtype=dtype)

                b=float_to_bfp_tiled(t, mant_bits, epsilon, rounding_mode, device, tile_size , num_format)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                #print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))

                b=float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, tile_size , num_format)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                #print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))


def test_F_matmul_bfp():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bfp_args = {
        'num_format': 'bfp',
        'sparsity_num_format': 'fp32',
        'rounding_mode': 'stoc',
        'epsilon': 0.00000001,
        'mant_bits': 7,
        'weight_mant_bits': 15,
        'bfp_tile_size': 24,
        'bfp_block_size': 2,
        'in_sparsity': False,
        'w_sparsity': True,
        'grad_sparsity': False,
        'rearrange': False,
        'N': 0,
        'M': 0,
        'sparsity_frac': 0.50,
        'device': "cuda:0" if torch.cuda.is_available() else "cpu"
    }
    bfp_matmul = F_matmul_bfp(  num_format=bfp_args['num_format'], sparsity_num_format=bfp_args['sparsity_num_format'], mant_bits=bfp_args['mant_bits'], weight_mant_bits=bfp_args['weight_mant_bits'], 
                                bfp_block_size=bfp_args['bfp_block_size'], in_sparsity=bfp_args['in_sparsity'], w_sparsity=bfp_args['w_sparsity'], 
                                grad_sparsity=bfp_args['grad_sparsity'], rearrange=bfp_args['rearrange'], sparsity_frac=bfp_args['sparsity_frac'], device=bfp_args['device'])
    a = torch.tensor([[1, 2, 4, 8], [3, 7, 1, 2]]).to(device=device)
    b = torch.tensor([[2, 3, 5, 9], [3, 5, 9, 17], [4, 1, 8, 7], [6, 1, 3, 9]]).to(device=device)
    res = bfp_matmul(a, b)
    print(res)
    # assert res == torch.tensor([[50]]).to(device=device)

if __name__ == '__main__':
    test_F_matmul_bfp()
    # unittest.main(verbosity=2)
