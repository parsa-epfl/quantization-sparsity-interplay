"""
Copyright (c) 2024, Parallel Systems Architecture Laboratory (PARSA), EPFL & Machine Learning and Optimization Laboratory (MLO), EFPL
For the full license text, see transformers_hbfp_sparsity/LICENSE_HBFP.txt
"""

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