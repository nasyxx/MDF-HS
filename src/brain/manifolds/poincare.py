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
date     : Feb 28, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : poincare.py
project  : brain
license  : GPL-3.0+

Poincare model.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array
from functools import partial


@partial(jax.jit, inline=True)
def sqnorm(x: Array) -> Array:
    """Square norm of x."""
    return jnp.sum(x * x, axis=-1)


@partial(jax.jit, static_argnames=("eps",))
def mobius_add(u: Array, v: Array, eps: float = 1e-12) -> Array:
    r"""Mobius addition.

    Parameters
    ----------
    x : Array
        The first point.
    y : Array
        The second point.
    c : Array
        1/sqrt(c) is the Poincare ball radius.
    eps : float
        The epsilon to avoid division by zero.

    Returns
    -------
    Array
        The result of the mobius addition.
    """
    v = v + eps
    uv = 2 * jnp.dot(u, v)
    sqnorm_u = sqnorm(u)
    sqnorm_v = sqnorm(v)
