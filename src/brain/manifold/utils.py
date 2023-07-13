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
date     : Feb 27, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : brain
license  : GPL-3.0+

Utils
"""
# Standard Library
from functools import partial

# Types
from jaxtyping import Array

# JAX
import jax
import jax.numpy as jnp


EPS = 1e-15
BEPS = 1e-7  # Ball boundary eps


@partial(jax.jit, inline=True)
def safe_sqrt(x: Array) -> Array:
    """Safe sqrt."""
    return jnp.sqrt(jnp.maximum(x, EPS))


@partial(jax.jit, static_argnames=("axis", "keepdims"), inline=True)
def sq_l2norm(x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
    """Square norm."""
    if (xx := x * x).ndim == 0:
        return xx
    return jnp.sum(xx, axis=axis, keepdims=keepdims)


@partial(jax.jit, inline=True)
def safe_arctanh(x: Array) -> Array:
    """Safe artanh."""
    return jnp.arctanh(jnp.clip(x, -1 + 1e-7, 1 - 1e-7))


@partial(jax.jit, static_argnames=("axis",), inline=True)
def dot(x: Array, y: Array, axis: int | None = -1) -> Array:
    """Inner product with batch as the first axis."""
    if (xy := x * y).ndim == 0:
        return xy
    return jnp.sum(xy, axis=axis, keepdims=True)
