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
filename : input.py
project  : brain
license  : GPL-3.0+

Input dataset.
"""
# Standard Library
from collections.abc import Iterator
from dataclasses import asdict
from itertools import chain
import pickle
from functools import cache
from pathlib import Path
from config import Conf

# Types
from jaxtyping import Array

# Math
import numpy as np

# JAX
import jax
import jax.numpy as jnp

# Local
from .graph import Brain
from .utils import batchify, unbatchify
from brain.manifold import poincare
from smile_config import from_dict

P = Path | str


def load_fmri(path: Path) -> Array:
    """Load fmri data."""
    return jnp.asarray(np.loadtxt(path, dtype=np.float32))


def read_fmri(path: P) -> Array:
    """Read fmri data."""
    if isinstance(path, str):
        path = Path(path)
    fmri = jnp.stack(
        list(
            map(load_fmri, sorted((path / "nonzero_fmri_average_signal").glob("*.txt")))
        )
    ).T
    assert fmri.shape == (148, 191)
    return fmri


def read_fiber(path: P) -> Array:
    """Read fiber data."""
    if isinstance(path, str):
        path = Path(path)
    fiber = jnp.asarray(
        np.loadtxt(path / "nonzero_common_fiber_matrix.txt", dtype=np.float32)
    )
    assert fiber.shape == (148, 148)
    return jnp.log10(fiber + 1)


def read_feature(path: P) -> Array:
    """Read feature data."""
    if isinstance(path, str):
        path = Path(path)
    feature = jnp.asarray(
        np.delete(
            np.delete(
                np.loadtxt(path / "pcc_fmri_feature_matrix_0.txt", dtype=np.float32),
                [41, 116],
                axis=1,
            ),
            [41, 116],
            axis=0,
        )
    )
    assert feature.shape == (148, 148)
    return feature


@cache
def read_label(path: P) -> tuple[set[str], set[str]]:
    """Read label data."""
    if isinstance(path, str):
        path = Path(path)
    with (
        (path / "NC_used.txt").open() as nc,
        (path / "MCI_used.txt").open() as mci,
    ):
        return set(nc.read().splitlines()), set(mci.read().splitlines())


def read_brain(path: P) -> Brain:
    """Read brain data."""
    if isinstance(path, str):
        path = Path(path)
    fmri = read_fmri(path)
    fmri_pair = jax.vmap(lambda x: x - fmri)(fmri)
    nc, mci = read_label(path / "..")
    if path.stem in nc:
        label = jnp.array(0, dtype=jnp.int32)
    elif path.stem in mci:
        label = jnp.array(1, dtype=jnp.int32)
    else:
        raise ValueError(f"Unknown label for {path.stem}")

    return Brain(
        fmri,  # fmri,
        fmri_pair,  # fmri_pair
        read_fiber(path),  # adj_s
        read_feature(path),  # p
        label,  # label
        jnp.array(0, dtype=jnp.int32),  # group
    )


def split_brain(data: Brain, group: int) -> Iterator[Brain]:
    """Split a brain into 4."""
    import ipdb;ipdb.set_trace()
    yield from map(
        lambda fmri: Brain(
            fmri,  # fmri,
            jax.vmap(lambda x: x - fmri)(fmri),  # fmri_pair
            data.adj_s,  # adj_s
            jnp.corrcoef(fmri),  # p
            data.label,  # label
            jnp.array(group, dtype=jnp.int32),  # group
        ),
        map(lambda i: data.fmri[:, i : i + 45], range(0, 45 * 4, 45)),
    )


def read_all_brain(conf: Conf, overwrites: dict | None = None) -> list[Brain]:
    """Read all brain data."""
    if overwrites is not None:
        conf = from_dict(conf.__class__, asdict(conf) | overwrites)
    path = Path(conf.data_path)

    sp = conf.split and "-split" or ""
    nm = conf.norm and "-norm" or ""
    mm = f"-{conf.manifold}"
    cc = f"-{conf.c}"
    if (pkl_path := (path / f"brain{sp}{nm}{mm}{cc}.pkl")).exists() and (not conf.reload):
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    res = list(map(read_brain, filter(lambda x: x.is_dir(), sorted(path.iterdir()))))
    import ipdb;ipdb.set_trace()
    if conf.split:
        conf.log("Split brains into 4")
        res = list(chain.from_iterable(map(split_brain, res, range(len(res)))))

    if conf.manifold == "poincare" or conf.manifold == "p":
        conf.log("Project to Poincare ball")
        brains = []
        c = jnp.asarray(conf.c, dtype=res[0].fmri.dtype)
        for b in res:
            nfmri = poincare.proj(poincare.expmap0(b.fmri, c), c)
            nfmri_pair = jax.vmap(
                lambda x: poincare.mobius_add(x, -nfmri, c)  # noqa: B023
            )(nfmri)
            n_p = poincare.proj(poincare.expmap0(b.p, c), c)
            brains.append(
                b._replace(
                    fmri=nfmri,
                    fmri_pair=nfmri_pair,
                    p=n_p,
                )
            )
            del nfmri, nfmri_pair, n_p
        res = brains
        del brains

    if conf.norm:
        conf.log("Normalize")
        brain = batchify(res)
        res = unbatchify(
            Brain(
                jax.nn.standardize(brain.fmri, axis=(0, 1)),
                jax.nn.standardize(brain.fmri_pair, axis=(0, 2)),
                jax.nn.standardize(brain.adj_s, axis=(0, 1, 2)),
                jax.nn.standardize(brain.p, (0, 1, 2)),
                brain.label,
                brain.group,
            )
        )

    with pkl_path.open("wb") as f:
        pickle.dump(res, f)
    return res
