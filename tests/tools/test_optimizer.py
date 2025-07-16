import sys
import os
# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import torch
from tools.optimizer import MomentumOptimizer, MomentumOptimizerTorch


class TestMomentumOptimizer():
    def test_compute_grad(self):
        optimizer = MomentumOptimizer()
        theta = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        new_theta = optimizer.move_in_grad(theta, grad, "min")
        assert new_theta.shape == theta.shape
        new_theta2 = optimizer.move_in_grad(theta, grad, "max")
        assert new_theta2.shape == theta.shape

        # Check that the new theta is different from the old theta
        assert np.all(new_theta != theta)
        assert np.all(new_theta2 != theta)

    def test_compute_grad_torch(self):
        # Implement function that returns the following gradient:
        # np.array([0.1, 0.2, 0.3])
        def f(x):
            return 0.1 * x[0] + 0.2 * x[1] + 0.3 * x[2]

        theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = MomentumOptimizerTorch([theta])
        optimizer.zero_grad()
        loss = f(theta)
        loss.backward()
        optimizer.step()