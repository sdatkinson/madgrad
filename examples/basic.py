# File: basic.py
# Created Date: Tuesday March 30th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Simple demonstration on a linear regression problem
"""

from argparse import ArgumentParser
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, value_and_grad, vmap
from jax.experimental.optimizers import sgd
from madgrad import madgrad
from tqdm import tqdm


def model(params, x):
    a, b = params
    return a * x + b


def loss(params, data):
    x, y = data
    preds = model(params, x)
    return jnp.mean(jnp.square(preds - y))


def main(args):
    x = jnp.linspace(0.0, 1.0, 5)
    a_true, b_true = 1.2, 3.2
    y = model((a_true, b_true), x)
    data = (x, y)

    init_fun, update_fun, get_params = madgrad()

    state = init_fun((jnp.array(0.0), jnp.array(0.0)))

    @jit
    def step(i, state):
        f, g = value_and_grad(loss)(get_params(state), data)
        return f, update_fun(i, g, state)

    losses = []
    for i in tqdm(range(1_000)):
        iter_loss, state = step(i, state)
        losses.append(iter_loss)

    params = get_params(state)
    print(params)

    plt.figure()
    plt.semilogy(losses)

    plt.figure()
    plt.plot(y, model(params, x), ".")
    plt.plot(plt.xlim(), plt.xlim(), "--")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    main(parser.parse_args())
