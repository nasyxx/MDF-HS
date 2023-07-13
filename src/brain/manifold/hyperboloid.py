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
date     : Feb 26, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : manifold.py
project  : brain
license  : GPL-3.0+

Hyerboloid (Lorentz or Minkowski) manifold.

x0^2 - x1^2 - ... - xd^2 = K

c = 1 / K is the hyperbolic curvature.
"""
# Standard Library
from functools import partial

# Types
from jaxtyping import Array

# JAX
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("keepdim",))
def minkowski_dot(x: Array, y: Array, keepdim: bool = True) -> Array:
    """Minkowski dot product."""
    res = jnp.sum(x * y, axis=-1) - 2 * x[..., 0] * y[..., 0]
    if keepdim:
        return res[None]
    return res


@partial(jax.jit, static_argnames=("",))
def minkowski_norm(x: Array, keepdim: bool = True) -> Array:
    """Minkowski norm."""
    res = jnp.sqrt(minkowski_dot(x, x, keepdim=keepdim))
    if keepdim:
        return res[None]
    return res


@jax.jit
def sqdist(x: Array, y: Array, c: Array) -> Array:
    """Squared distance."""
    k = 1 / c
    prod = minkowski_dot(x, y)
    theta = jnp.arccosh(-prod / k)
    return k * (theta**2)


@jax.jit
def proj(x: Array, c: Array) -> Array:
    """Project x on the manifold."""
    k = 1 / c
    return jnp.concatenate(
        [jnp.sqrt(k + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)), x[..., 1:]],
        axis=-1,
    )


@jax.jit
def proj_tan(u: Array, p: Array) -> Array:
    """Project u on the tangent space of p."""
    return jnp.concatenate(
        [
            jnp.sum(u[..., 1:] * p[..., 1:], axis=-1, keepdims=True) / p[..., 0],
            u[..., 1:],
        ],
        axis=-1,
    )


@jax.jit
def proj_tan0(u: Array) -> Array:
    """Project u on the tangent space of u."""
    return u.at[..., 0].set(0)


@jax.jit
def expmap(u: Array, p: Array, c: Array) -> Array:
    """Exponential map."""
    k = 1 / c
    theta = minkowski_norm(u)
    return jnp.concatenate(
        [
            jnp.cosh(theta) * p[..., 0] - k * jnp.sinh(theta) * u[..., 0],
            jnp.sinh(theta) * p[..., 1:] / theta,
        ],
        axis=-1,
    )
