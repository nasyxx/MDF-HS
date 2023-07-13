#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Feb 24, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : brain
license  : GPL-3.0+

Utils
"""
# Standard Library
from collections.abc import Callable
from functools import partial
from config import Conf

# Types
from jaxtyping import Array

# JAX
import jax
import jax.numpy as jnp
import haiku as hk
from typing import cast


@jax.jit
def safe_sqrt(x: Array) -> Array:
    """Safe sqrt."""
    return jnp.sqrt(jnp.maximum(x, 1e-12))


@partial(jax.jit, static_argnums=(1,))
def l2norm_without_sqrt(x: Array, axis: int = -1) -> Array:
    """L2 norm without sqrt."""
    return jnp.sum(x**2, axis=axis)


def get_act(n: str, **extra: dict[str, float]) -> Callable[[Array], Array]:
    """Get activation function."""
    return cast(Callable[[Array], Array], {
        "relu": jax.nn.relu,
        "leaky_relu": partial(
            jax.nn.leaky_relu, negative_slope=extra.get("leaky_relu_slope", 0.2)
        ),
    }[n])
