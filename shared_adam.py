"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

