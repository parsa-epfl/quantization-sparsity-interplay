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
from .bfp_ops import float_to_bfp_tiled, unpack_bfp_args

_bfp_optims = {}
def _gen_bfp_optim(optim, name):
    class BFPOptim(optim):
        """
        Wrap the model's original optimizer in BFP
        Perform the original optimizer's  update function in fp32
        Convert the weights to two BFP formats: One with wide and another with narrow mantissas.
            Wide weights are used in future weight updates
            Narrow weights are used in forward and backward passes.
        """
        def __init__(self, *args, **kwargs):
            self.bfp_args = unpack_bfp_args(kwargs)
            super().__init__(*args, **kwargs)

        def step(self, *args, **kwargs):
            if self.bfp_args['num_format'] == 'fp32':
                return super().step(*args, **kwargs)

            # Move wide weights to p.data
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    # Init step, just constraint pdata
                    if 'shadow_p' not in state:
                        p.data.copy_(float_to_bfp_tiled(p.data, sgd_update=True, **self.bfp_args))
                    else:
                        shadow_p = state['shadow_p']
                        p.data.copy_(shadow_p)

            # Apply step
            loss = super().step(*args, **kwargs)

            # Move wide weights to shadow_p and move extracted weights to p.data
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'shadow_p' not in state:
                        state['shadow_p'] = torch.zeros_like(p.data)

                    shadow_p = state['shadow_p']
                    shadow_p.copy_(float_to_bfp_tiled(p.data, sgd_update=True, **self.bfp_args))
                    p.data.copy_(float_to_bfp_tiled(p.data, **self.bfp_args))

            return loss

    BFPOptim.__name__ = "BFP" + name
    return BFPOptim


def get_bfp_optim(optim, name):
    if name not in _bfp_optims:
        _bfp_optims[name] = _gen_bfp_optim(optim, name)

    return _bfp_optims[name]