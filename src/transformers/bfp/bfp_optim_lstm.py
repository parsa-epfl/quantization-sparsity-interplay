"""
Copyright (c) 2024, Parallel Systems Architecture Laboratory (PARSA), EPFL & Machine Learning and Optimization Laboratory (MLO), EFPL
For the full license text, see transformers_hbfp_sparsity/LICENSE_HBFP.txt
"""
import torch
import math
from .bfp_ops import float_to_bfp_blocked, unpack_bfp_args
from .bfp_util import get_bfp_args

required=object()

class BFPAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.bfp_args = get_bfp_args()
        #print('------------------- bfpadam init -----------------------')

        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        #print('------------------- bfpadam step -----------------------')

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if self.bfp_args['num_format'] == 'fp32':
                    #print('....... in if')
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                elif self.bfp_args['num_format'] == 'bfp':
                    #print('....... in else')
                    updated_value = float_to_bfp_blocked(p.data.addcdiv_(exp_avg, denom, value=-step_size), sgd_update=True, **self.bfp_args)

                    p.data.copy_(updated_value.data)
                else:
                    raise NotImplementedError('NumFormat not implemented')



        return loss


