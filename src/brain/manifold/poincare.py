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
filename : poincare.py
project  : brain
license  : GPL-3.0+

Poincare ball model.

B^{n}_{k} = \{ x \in R^{n} | ||x|| < k \}

let x0^2 + x1^2 + ... + xd^2 < 1/c, where 1/sqrt(c) is the
Poincare ball radius.   And the curvature k = 1/c.
"""
# Standard Library
from functools import partial

# Types
from jax.typing import ArrayLike
from typing import cast

# JAX
import jax
import jax.numpy as jnp

# Local
from .base import Manifold
from .utils import sq_l2norm, safe_arctanh, safe_sqrt, dot, EPS, BEPS


Array = jax.Array


@partial(jax.jit, inline=True)
def radius(c: ArrayLike) -> Array:
    r"""Radius of the Poincare ball."""
    return (1 - BEPS) / jnp.sqrt(c)


@partial(jax.jit, inline=True)
def mobius_add(x: Array, y: Array, c: ArrayLike) -> Array:
    """Mobius addition."""
    x_sqnorm = c * sq_l2norm(x, axis=-1, keepdims=True)
    y_sqnorm = c * sq_l2norm(y, axis=-1, keepdims=True)
    xy = c * dot(x, y)
    num = (1 + 2 * xy + y_sqnorm) * x + (1 - x_sqnorm) * y
    denom = 1 + 2 * xy + x_sqnorm * y_sqnorm
    return num / (denom + EPS)


@partial(jax.jit, inline=True)
def mobius_scalar_mul(r: ArrayLike, x: Array, c: ArrayLike) -> Array:
    """Mobius scalar multiplication."""
    r = jnp.asarray(r, dtype=x.dtype)
    # if jnp.all(r == 0):
    #     return jnp.zeros_like(x)
    ra = radius(c)
    x_norm = safe_sqrt(sq_l2norm(x, axis=-1, keepdims=True))
    return ra * jnp.tanh(r * safe_arctanh(x_norm / ra)) * (x / (x_norm + EPS))


@partial(jax.jit, inline=True)
def mobius_matvec(m: Array, x: Array, c: ArrayLike) -> Array:
    """Mobius matrix-vector multiplication."""
    sqrt_c = jnp.sqrt(c)
    x_norm = safe_sqrt(sq_l2norm(x, axis=-1, keepdims=True))
    mx = m @ x
    mx_norm = safe_sqrt(sq_l2norm(mx, axis=-1, keepdims=True))
    res_c = (
        jnp.tanh(mx_norm / x_norm * safe_arctanh(sqrt_c * x_norm))
        * mx
        / (mx_norm * sqrt_c)
    )
    return cast(Array, jnp.where((mx == 0).prod(-1, keepdims=True), 0, res_c))


@partial(jax.jit, inline=True)
def sqdist(x: Array, y: Array, c: ArrayLike) -> Array:
    """Squared distance between two points."""
    sqrt_c = safe_sqrt(c)
    distc = safe_arctanh(
        safe_sqrt(c * sq_l2norm(mobius_add(-x, y, c, axis=-1), axis=-1, keepdims=False))
    )
    return (distc * 2 / sqrt_c) ** 2


@partial(jax.jit, inline=True)
def lambda_x(x: Array, c: ArrayLike) -> Array:
    """Lambda function for x."""
    return 2 / jnp.maximum(1 - c * sq_l2norm(x, axis=-1, keepdims=True), EPS)


@partial(jax.jit, inline=True)
def proj(x: Array, c: ArrayLike) -> Array:
    """Project a point to the Poincare ball."""
    r = radius(c)
    norm = safe_sqrt(sq_l2norm(x, axis=-1, keepdims=True))
    return cast(
        Array,
        jnp.where(
            norm > r,
            x / norm * r,
            x,
        ),
    )


@partial(jax.jit, donate_argnums=(1, 2), inline=True)
def proj_tan(u: Array, _p: Array, _c: ArrayLike) -> Array:
    """Project a tangent vector to the Poincare ball."""
    return u


@partial(jax.jit, donate_argnums=(1,), inline=True)
def proj_tan0(u: Array, _c: ArrayLike) -> Array:
    """Project a tangent vector to the Poincare ball."""
    return u


@partial(jax.jit, inline=True)
def expmap(u: Array, p: Array, c: ArrayLike) -> Array:
    """Exponential map u at p."""
    sqrt_c = safe_sqrt(c)
    unorm = safe_sqrt(sq_l2norm(u, axis=-1, keepdims=True))
    return mobius_add(
        p,
        jnp.tanh(sqrt_c * lambda_x(p, c) * unorm / 2) * u / (sqrt_c * unorm),
        c,
    )


@partial(jax.jit, inline=True)
def expmap0(u: Array, c: ArrayLike) -> Array:
    """Exponential map u at p."""
    sqrt_c = safe_sqrt(c)
    unorm = safe_sqrt(sq_l2norm(u, axis=-1, keepdims=True))
    return jnp.tanh(sqrt_c * unorm) * u / (sqrt_c * unorm)


@partial(jax.jit, inline=True)
def logmap(u: Array, p: Array, c: ArrayLike) -> Array:
    """Logarithmic map u at p."""
    sqrt_c = jnp.sqrt(c)
    sub = mobius_add(-u, p, c)
    sub_norm = safe_sqrt(sq_l2norm(sub, axis=-1, keepdims=True))
    return (
        (2 / (sqrt_c * lambda_x(u, c)))
        * safe_arctanh(sqrt_c * sub_norm)
        * (sub / sub_norm)
    )


@partial(jax.jit, inline=True)
def logmap0(u: Array, c: ArrayLike) -> Array:
    """Logarithmic map u at p."""
    sqrt_c = safe_sqrt(c)
    sqrt_norm = safe_sqrt(sq_l2norm(u, axis=-1, keepdims=True))
    return u / sqrt_c * safe_arctanh(sqrt_c * sqrt_norm) / sqrt_norm


@partial(jax.jit, inline=True)
def egrad2rgrad(p: Array, dp: Array, c: ArrayLike) -> Array:
    """Convert Euclidean gradient to Riemannian gradient."""
    return dp / lambda_x(p, c) ** 2


@partial(jax.jit, static_argnames=("keepdims",), inline=True)
def inner(x: Array, c: ArrayLike, u: Array, v: Array, keepdims: bool = False) -> Array:
    """Poincare inner product at point x with radius c."""
    return lambda_x(x, c) ** 2 * (u * v).sum(axis=-1, keepdims=keepdims)


@partial(jax.jit, static_argnames=("axis",), inline=True)
def gyration(u: Array, v: Array, w: Array, c: ArrayLike, axis: int = -1) -> Array:
    """Gyration of vectors u, v, w."""
    u2 = dot(u, u, axis=axis)
    v2 = dot(v, v, axis=axis)
    uv = dot(u, v, axis=axis)
    uw = dot(u, w, axis=axis)
    vw = dot(v, w, axis=axis)
    c2 = c ** 2
    a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + c2 * u2 * v2
    return w + 2 * (a * u + b * v) / jnp.maximum(d, EPS)


@partial(jax.jit, inline=True)
def ptransp(x: Array, y: Array, u: Array, c: ArrayLike) -> None:
    """Parallel transport of u from x to y."""
    lx = lambda_x(x, c)
    ly = lambda_x(y, c)
    return gyration(y, -x, u, c) * lx / ly


poincare = Manifold(
    sqdist,
    inner,
    proj,
    proj_tan,
    proj_tan0,
    expmap,
    expmap0,
    logmap,
    logmap0,
    mobius_add,
    mobius_matvec,
    mobius_scalar_mul,
    ptransp,
    lambda_x,
    egrad2rgrad,
)
