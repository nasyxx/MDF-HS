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
date     : Feb 23, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : brain
license  : GPL-3.0+

Utils for brain dataloader.
"""
# Types
from collections.abc import Iterator
from config import Conf
from jaxtyping import Array, PyTree
from .graph import Brain

# JAX
import jax
import jax.numpy as jnp
from sklearn.model_selection import StratifiedGroupKFold


@jax.jit
def normalize(data: Array) -> Array:
    """Normalize the data."""
    return (data - data.mean()) / data.std()


@jax.jit
def batchify(data: PyTree) -> PyTree:
    """Batchify the data."""
    return jax.tree_map(lambda *xs: jnp.stack(xs), *data)


def unbatchify(data: Brain) -> list[Brain]:
    """Unbatchify the data."""
    return list(map(lambda x: Brain(*x), zip(*data, strict=True)))


def fold(
    data: list[PyTree], k: int, seed: int, group: bool = True
) -> Iterator[tuple[list[PyTree], list[PyTree]]]:
    """Fold the data."""
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    gp = (jnp.arange(len(data)), batchify(data).group)[group]
    for train_idx, test_idx in skf.split(data, batchify(data).label, gp):
        yield (
            list(map(lambda ii: data[ii], train_idx)),
            list(map(lambda ii: data[ii], test_idx)),
        )


def train_test_group_split(
    data: list[PyTree], k: int, seed: int, group: bool = True
) -> tuple[list[PyTree], list[PyTree]]:
    """Split the data into train and test."""
    return next(fold(data, k, seed, group))
