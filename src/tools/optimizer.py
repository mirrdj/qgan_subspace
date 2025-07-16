# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimizer file"""

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from config import CFG
from tools.data.data_reshapers import _flatten, _unflatten


class MomentumOptimizer:
    """Gradient descent with momentum, given an objective function to be minimized.

    The update formula is from:
    "On the importance of initialization and momentum in deep learning
    http://proceedings.mlr.press/v28/sutskever13.pdf"

    v_{t+1} = miu * v_{t} - eta * grad(theta_t)
    theta_{t+1} = theta_{t} + v_{t+1}

    Args:
        eta: learning rate > 0
        miu: momentum coefficient [0,1]
    """

    def __init__(self, eta: float = CFG.l_rate, miu: float = CFG.momentum_coeff):
        self.miu: float = miu
        self.eta: float = eta
        self.v = None

    def move_in_grad(self, theta: np.ndarray, grad_list: np.ndarray, min_or_max: str) -> np.ndarray:
        grad_list = _flatten(grad_list)
        theta_tmp = _flatten(theta)

        if min_or_max == "min":
            sign = -1.0
        elif min_or_max == "max":
            sign = 1.0
        else:
            raise ValueError("min_or_max must be either 'min' or 'max'")

        if self.v is None:
            self.v = [sign * self.eta * g for g in grad_list]
        else:
            self.v = [self.miu * v + sign * self.eta * g for v, g in zip(self.v, grad_list)]

        new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]
        new_theta = _unflatten(new_theta, theta)

        return new_theta


class MomentumOptimizerTorch(Optimizer):
    def __init__(self, params, eta: float = CFG.l_rate, miu: float = CFG.momentum_coeff, min_or_max: str = "min"):
        if min_or_max == "min":
            self.sign = -1.0
        elif min_or_max == "max":
            self.sign = 1.0
        else:
            raise ValueError("min_or_max must be either 'min' or 'max'")


        defaults = dict(lr=eta, momentum=miu)
        super(MomentumOptimizerTorch, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            mu = group['momentum']
            eta = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                param_state = self.state[p]

                # Initialize velocity buffer if not present
                if 'velocity' not in param_state:
                    # Start velocity at zero (same shape as param)
                    param_state['velocity'] = torch.zeros_like(p.data)

                v = param_state['velocity']

                # Update velocity: v_{t+1} = mu * v_t - eta * grad
                v.mul_(mu).add_(grad, alpha=-eta)

                # Update parameter: theta_{t+1} = theta_t + v_{t+1}
                p.data.add_(v)

        return loss
