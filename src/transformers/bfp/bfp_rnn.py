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

"""
This file adds BFP functionality to LSTMs
We do the following:
Rewrite LSTMCell (BFPLSTMCell) to use BFP [done]
Rewrite StackedRNN (BFPStacked) to use the modified LSTMCell (BFPLSTMCell) []
Rewrite AutogradRNN (BFPAutogradRNN) to use the modified StackedRNN (BFPStackedRNN) []
Rewrite RNNBase (BFPRNNBase) to use BFPAutogradRNN
Rewrite LSTM (BFPLSTM) to use BFPRNNBase
There is probably a better way to do this, but I don't have the time right now.
"""
from .bfp_ops import _get_bfp_op, unpack_bfp_args
from torch.nn.modules.rnn import RNNBase
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import torch
'''
def lstm_cell(input: Tensor, hidden: Tuple[Tensor, Tensor], w_ih: Tensor,
              w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy
'''
def BFPLSTMCell(input, hidden, w_ih, w_hh, linear_op=None, b_ih=None, b_hh=None):
    '''
    if input.is_cuda:
        if linear_op is None:
            igates = F.linear(input, w_ih)
            hgates = F.linear(hidden[0], w_hh)
        else:
            igates = linear_op(input, w_ih)
            hgates = linear_op(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
    '''

    hx, cx = hidden
    if linear_op is None:
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    else:
        if b_ih is not None:
            ih_gate = linear_op(input, w_ih) + b_ih
        else:
            ih_gate = linear_op(input, w_ih)

        if b_hh is not None:
            bh_gate = linear_op(hx, w_hh) + b_hh
        else:
            bh_gate = linear_op(hx, w_hh)

        gates = ih_gate + bh_gate

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def BFPAutogradRNN(mode, input_size, hidden_size, num_layers=1,
                   batch_first=False,
                   dropout=0, train=True, bidirectional=False,
                   variable_length=False,
                   _flat_weights=None, **kwargs):

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        bfp_args = unpack_bfp_args(kwargs)
        if bfp_args['num_format'] == 'bfp':
            linear_op = _get_bfp_op(F.linear, 'linear', bfp_args)
        else:
            linear_op = None

        def cell_pass(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
            return BFPLSTMCell(input, hidden, w_ih, w_hh, linear_op,
                                 b_ih, b_hh)
        cell = cell_pass
    elif mode == 'GRU':
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    rec_factory = variable_recurrent_factory if variable_length else Recurrent

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and not variable_length:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and not variable_length:
            output = output.transpose(0, 1)

        return output, nexth

    return forward



class BFPRNNBase(RNNBase):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, proj_size=0,
                 **kwargs):
        super().__init__(mode, input_size, hidden_size,
                         num_layers, bias, batch_first,
                         dropout, bidirectional,proj_size)
        self.bfp_args = unpack_bfp_args(kwargs)


    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            assert isinstance(input, Tensor)
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        assert isinstance(input, Tensor)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx[0], batch_sizes)
        self.check_forward_args(input, hx[1], batch_sizes)
        _impl = BFPAutogradRNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            variable_length=is_packed,
            _flat_weights=self._flat_weights,
            **self.bfp_args
        )
        ## TODO:  output, hidden = func(input, self.all_weights, hx, batch_sizes)
        result = _impl(input, self.all_weights, hx)

        output: Union[Tensor, PackedSequence]
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

class BFPLSTM(BFPRNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)