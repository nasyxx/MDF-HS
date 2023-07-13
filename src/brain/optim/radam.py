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
filename : radam.py
project  : brain
license  : GPL-3.0+

Riemannian Adam.
"""
# Standard Library
from collections.abc import Callable, Iterable, Mapping

# Types
from jax.typing import ArrayLike
from jaxtyping import Array
from typing import Any, TypeAlias

# JAX
import chex
import jax
import jax.numpy as jnp
import optax
from optax import (
    GradientTransformation,
    ScaleByAdamState,
    bias_correction,
    update_moment,
)
from optax._src import utils

# Local
from brain.manifold import Manifold

PArray: TypeAlias = Array | Iterable["PArray"] | Mapping[Any, "PArray"]


def scale_by_radam(
    manifold: Manifold,
    b1: float = 0.9,
    b2: float = 0.999,
    c: float = 1.0,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
) -> GradientTransformation:
    """Scale by RAdam."""

    @jax.jit
    def rupdate_moment_per_elem_norm(
        updates: PArray, params: PArray, moments: PArray, decay: ArrayLike
    ) -> PArray:
        """Update moments per element norm."""
        return jax.tree_map(
            lambda g, p, t: (1 - decay)
            * manifold.inner(
                p, jnp.asarray(c, dtype=g.dtype), g, g, True  # noqa: FBT003
            )
            + t * decay,
            updates,
            params,
            moments,
        )

    def init_fn(params: PArray) -> ScaleByAdamState:
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(
        updates: PArray, state: ScaleByAdamState, params: PArray | None = None
    ) -> tuple[PArray, ScaleByAdamState]:
        updates = jax.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        updates = jax.tree_map(
            lambda p, u: manifold.egrad2rgrad(p, u, c), params, updates
        )
        mu = update_moment(updates, state.mu, b1, 1)
        nu = rupdate_moment_per_elem_norm(updates, params, state.nu, b2)
        count_inc = optax.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)  # type: ignore[arg-type]


def apply_updates(
    manifold: Manifold, c: float, params: PArray, updates: PArray
) -> PArray:
    """Apply update."""
    return jax.tree_map(
        lambda p, u: manifold.proj(manifold.expmap(u, p, c), c),  # type: ignore[arg-type]  # noqa: E501
        params,
        updates,
    )
