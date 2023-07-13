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
filename : dataset.py
project  : brain
license  : GPL-3.0+

Dataset for brain.
"""
# Standard Library
from functools import partial
import math
from collections.abc import Iterator
from operator import getitem
from typing import cast

# JAX
import jax

# Config
from config import Conf

# Local
from .graph import Brain
from .utils import batchify
from sklearn.model_selection import train_test_split


class Dataset:
    """Dataset for brain."""

    def __init__(self, data: list[Brain], conf: Conf) -> None:
        """Initialize the dataset."""
        self.data = data
        self.conf = conf
        self.rng = jax.random.PRNGKey(conf.seed)

    def __getitem__(self, index: int) -> Brain:
        """Get the brain."""
        return self.data[index]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)

    def new_rng(self) -> None:
        """Get a new rng."""
        self.rng, _ = jax.random.split(self.rng)

    def loader(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        *,
        new_rng: bool = False,
    ) -> Iterator[Brain]:
        """Get the loader."""
        if new_rng:
            self.new_rng()
        if shuffle:
            idxs = jax.random.permutation(self.rng, len(self.data))
            data = cast(list[Brain], list(map(partial(getitem, self.data), idxs)))
        else:
            data = self.data
        if batch_size is None:
            yield batchify(data)
            return

        nc = list(filter(lambda x: (x.label == 0).all(), data))
        mci = list(filter(lambda x: (x.label == 1).all(), data))

        if len(nc) > len(mci):
            ids = jax.random.permutation(self.rng, len(mci))[: len(nc) - len(mci)]
            mci += cast(list[Brain], list(map(partial(getitem, mci), ids)))
        if len(mci) > len(nc):
            ids = jax.random.permutation(self.rng, len(nc))[: len(mci) - len(nc)]
            nc += cast(list[Brain], list(map(partial(getitem, nc), ids)))

        for ii in range(0, len(nc), batch_size):
            yield batchify(
                nc[ii : ii + batch_size // 2] + mci[ii : ii + batch_size // 2]
            )

    def __repr__(self) -> str:
        """Get the representation of the dataset."""
        return f"Dataset({len(self.data)} brains)"
