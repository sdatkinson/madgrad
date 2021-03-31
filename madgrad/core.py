# File: core.py
# Created Date: Tuesday March 30th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

import jax.numpy as jnp
from jax.experimental.optimizers import Optimizer, make_schedule, optimizer


@optimizer
def madgrad(step_size=0.01, momentum=0.1, epsilon=1.0e-8):
    """
    Implementation of MADGRAD (Defazio and Jelassi, arXiv:2101.11075)
    """
    step_size = make_schedule(step_size)
    momentum = make_schedule(momentum)

    def init(x):
        s = jnp.zeros_like(x)
        nu = jnp.zeros_like(x)
        z = jnp.zeros_like(x)
        x0 = x
        return x, s, nu, z, x0

    def update(i, g, state):
        x, s, nu, z, x0 = state
        lbda = step_size(i) * jnp.sqrt(i + 1)
        s = s + lbda * g
        nu = nu + lbda * g * g
        z = x0 - s / (jnp.power(nu, 1.0 / 3.0) + epsilon)
        x = (1 - momentum(i)) * x + momentum(i) * z

        return x, s, nu, z, x0

    def get_params(state):
        x, s, nu, z, x0 = state
        return x

    return Optimizer(init, update, get_params)
