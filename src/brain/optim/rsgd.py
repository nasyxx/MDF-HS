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
date     : Mar  7, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : rsgd.py
project  : brain
license  : GPL-3.0+

Riemannian Stochastic Gradient Descent.
"""
from jax.typing import ArrayLike
import jax
from jax import Array
import jax.numpy as jnp
import optax
from optax import GradientTransformation, EmptyState, Params, Updates
from optax._src import utils

from brain.manifold import Manifold

from collections.abc import Callable, Iterable, Mapping


def scale_by_rsgd(manifold: Manifold, c: ArrayLike) -> GradientTransformation:

    def init_fn(_: Params) -> EmptyState:
        del _
        return EmptyState()

    def update_fn(
        updates: Updates, state: EmptyState, params: Params | None = None
    ) -> tuple[Params, EmptyState]:
        tc = params["model"].get("tc", 1.0)  # type: ignore
        c_ = tc * c
        updates = jax.tree_map(
            lambda u, p: manifold.egrad2rgrad(p, u, c_), updates, params
        )
        return updates, state

    return GradientTransformation(init_fn, update_fn)


def apply_rsgd_updates(
    manifold: Manifold,
    c: ArrayLike,
    updates: Updates,
    params: Params,
) -> Updates:
    """Apply Riemannian SGD update."""
    tc = params["model"].get("tc", 1.0)  # type: ignore
    c_ = tc * c
    return jax.tree_map(
        lambda u, p: manifold.proj(manifold.expmap(p, u, c_), c_), updates, params
    )
