##### Optimizer file


from tools.data_reshapers import _flatten, _unflatten


class MomentumOptimizer:
    """
    gradient descent with momentum, given an objective function to be minimized.
    the update formula from
    On the importance of initialization and momentum in deep learning
    http://proceedings.mlr.press/v28/sutskever13.pdf

    v_{t+1} = miu * v_{t} - learning_rate * grad(theta_t)
    theta_{t+1} = theta_{t} + v_{t+1}

    learning_rate:= learning rate > 0
    miu:= momentum coefficient [0,1]
    """

    def __init__(self, learning_rate=0.01, miu=0.9):
        self.miu = miu
        self.learning_rate = learning_rate
        self.v = None

    def compute_grad(self, theta, grad_list, min_or_max):
        grad_list = _flatten(grad_list)
        theta_tmp = _flatten(theta)

        if min_or_max == "min":
            if self.v is None:
                self.v = [-self.learning_rate * g for g in grad_list]
            else:
                self.v = [self.miu * v - self.learning_rate * g for v, g in zip(self.v, grad_list)]

            new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]
        else:
            if self.v is None:
                self.v = [self.learning_rate * g for g in grad_list]
            else:
                self.v = [self.miu * v + self.learning_rate * g for v, g in zip(self.v, grad_list)]

            new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]

        new_theta = _unflatten(new_theta, theta)

        return new_theta
