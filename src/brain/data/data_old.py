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
date     : Feb 13, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : data.py
project  : brain
license  : GPL-3.0+

Data for brain.
"""
# Standard Library
import pickle
from collections.abc import Iterator
from functools import cache
from glob import glob
from os.path import basename, dirname, exists, isdir, join
from random import choice

# Types
from jaxtyping import Array
from typing import NamedTuple

# Math
import numpy as np
from sklearn.model_selection import train_test_split

# JAX
import jax
import jax.numpy as jnp

# Config
from config import Conf


class Brain(NamedTuple):
    """Brain graph tuple.

    N: 148
    T: 191
    C: 1
    """

    fmri: Array  # NxNxT
    adj_s: Array  # NxN
    p: Array  # NxN
    label: Array  # C


def norm(x: Array) -> Array:
    """Normalize the data."""
    return (x - x.mean()) / x.std()


def logfiber(x: Array) -> Array:
    """Log the fiber."""
    return norm(jnp.log10(x + 1))


def load_node(path: str) -> Array:
    """Load node data."""
    return jnp.array(np.loadtxt(path))


@cache
def load_label_name(path: str) -> tuple[set[str], set[str]]:
    """Load true name."""
    with (
        open(join(path, "NC_used.txt")) as nc,
        open(join(path, "MCI_used.txt")) as mci,
    ):
        return set(nc.read().splitlines()), set(mci.read().splitlines())


def load_raw_data(path: str) -> Brain:
    """Load raw data from path."""
    nc, mci = load_label_name(dirname(path))
    if basename(path) in nc:
        label = jnp.array(0, dtype=jnp.int32)
    elif basename(path) in mci:
        label = jnp.array(1, dtype=jnp.int32)
    else:
        raise ValueError(f"{path} is not in NC or MCI.")

    fmri = norm(
        jnp.stack(
            list(map(load_node, glob(join(path, "nonzero_fmri_average_signal/*.txt"))))
        ).T
    )

    adj_s = logfiber(
        jnp.asarray(np.loadtxt(join(path, "nonzero_common_fiber_matrix.txt"))) + 1
    )

    p = jnp.array(
        np.delete(
            np.delete(
                np.loadtxt(join(path, "pcc_fmri_feature_matrix_0.txt")),
                [41, 116],
                axis=1,
            ),
            [41, 116],
            axis=0,
        )
    )

    assert adj_s.shape == (148, 148)
    assert fmri.shape == (148, 191)
    assert p.shape == (148, 148)

    return Brain(fmri, adj_s, p, label)


def preprocess(conf: Conf) -> tuple[list[Brain], list[Brain]]:
    """Preprocess data."""
    data = list(
        map(
            load_raw_data,
            filter(isdir, glob(join(conf.data_path, "Data_MIA_deepFusion/*"))),
        )
    )
    nc = list(filter(lambda x: x.label == 0, data))
    mci = list(filter(lambda x: x.label == 1, data))
    with open(join(conf.data_path, "nc.pkl"), "wb") as ncf:
        pickle.dump(nc, ncf)
    with open(join(conf.data_path, "mci.pkl"), "wb") as mcif:
        pickle.dump(mci, mcif)
    return nc, mci


def load_data(conf: Conf) -> tuple[list[Brain], list[Brain]]:
    """Load data."""
    if exists(join(conf.data_path, "nc.pkl")) and exists(
        join(conf.data_path, "mci.pkl")
    ):
        with (
            open(join(conf.data_path, "nc.pkl"), "rb") as ncf,
            open(join(conf.data_path, "mci.pkl"), "rb") as mcif,
        ):
            return pickle.load(ncf), pickle.load(mcif)

    return preprocess(conf)


def batch(data: list[Brain]) -> Brain:
    """Batch data."""
    return Brain(
        jnp.stack(list(map(lambda x: x.fmri, data))),
        jnp.stack(list(map(lambda x: x.adj_s, data))),
        jnp.stack(list(map(lambda x: x.p, data))),
        jnp.stack(list(map(lambda x: x.label, data))),
    )


class Dataset:
    """Dataset."""

    def __init__(
        self, nc: list[Brain], mci: list[Brain], batch_size: int | None, conf: Conf
    ) -> None:
        """Init."""
        self.conf = conf
        self.nc, self.mci = nc, mci
        self.allbrain = self.nc + self.mci
        self.key = jax.random.PRNGKey(conf.seed)
        self.batch_size = batch_size

    def __getitem__(self, index: int) -> Brain:
        """Get item."""
        i = index // 2
        if index % 2 == 0:
            return self.nc[i]
        if i > len(self.mci) - 1:
            return choice(self.mci)
        return self.mci[i]

    def __len__(self) -> int:
        """Get length."""
        return self.batch_size and len(self.nc) * 2 or len(self.allbrain)

    def __repr__(self) -> str:
        """Get representation."""
        return (
            f"NC: {batch(self.nc).fmri.shape}\n"
            f"MCI: {batch(self.mci).fmri.shape}\n"
            f"Length: {len(self)}\n"
            f"Overlap {len(self.nc) - len(self.mci)} MCI"
        )

    def loader(self, batch_size: int | None = None) -> Iterator[Brain]:
        """Get loader."""
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        if self.batch_size is None:
            yield batch(self.allbrain)
            return
        self.key, subkey = jax.random.split(self.key)
        idxs = jax.random.permutation(subkey, jnp.arange(len(self)))
        for idx in range(0, len(self), self.batch_size):
            ids = idxs[idx : idx + self.batch_size]
            yield batch(list(map(lambda i: self[i], ids)))


def gen_dataset(conf: Conf) -> tuple[Dataset, Dataset, Dataset]:
    """Get data."""
    nc, mci = load_data(conf)
    trnc, tenc = train_test_split(nc, test_size=0.2, random_state=conf.seed)
    trmci, temci = train_test_split(mci, test_size=0.2, random_state=conf.seed)
    return (
        Dataset(trnc, trmci, None, conf),
        Dataset(tenc, temci, None, conf),
        Dataset(nc, mci, None, conf),
    )
