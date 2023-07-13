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
date     : Mar  5, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : base.py
project  : brain
license  : GPL-3.0+

Some base.
"""
# Standard Library
import os, random

# Types
from jaxtyping import Array

# Math
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.decomposition import PCA

# Plot
import matplotlib as mpl
import matplotlib.pyplot as plt

# JAX
import jax
import jax.numpy as jnp

# Config
from config import Conf

mpl.rcParams["font.family"] = "serif"
mpl.use("PDF")


def auc_fn(label: Array, pred: Array) -> tuple[Array, Array, Array, Array]:
    """Compute the AUC."""
    # random.shuffle(lp := list(zip(label, pred, strict=True)))
    # label, pred = zip(*lp, strict=True)
    roc = roc_auc_score(label, pred)
    ps, rs, _ = precision_recall_curve(label, pred)
    pr = auc(rs, ps)
    return (
        jnp.asarray(ps, jnp.float32),
        jnp.asarray(rs, jnp.float32),
        jnp.asarray(roc, jnp.float32),
        jnp.asarray(pr, jnp.float32),
    )

def prf_fn(label: Array, pred: Array) -> tuple[Array, Array, Array]:
    """Compute the precision, recall, and f1."""
    return (
        jnp.asarray(precision_score(label, pred), jnp.float32),
        jnp.asarray(recall_score(label, pred), jnp.float32),
        jnp.asarray(f1_score(label, pred), jnp.float32),
    )


def diff_adj(base: Array, adjs: Array) -> Array:
    """Compute the difference of adj."""
    return adjs - base


def draw_heat(
    arr: Array, name: str, prefix: str = "n", vmin: int = 0, vmax: int = 5
) -> None:
    """Draw the heat map."""
    os.makedirs(f"results/{name}/figs", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.imshow(arr, cmap="gist_heat", norm="linear", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f"results/{name}/figs/{prefix}_heat.pdf")


def draw_auc(label: Array, pred: Array, name: str, prefix: str = "n") -> None:
    """Draw the AUC."""
    ps, rs, roc, pr = auc_fn(label, pred)
    fpr, tpr, _ = roc_curve(label, pred)

    os.makedirs(f"results/{name}/figs", exist_ok=True)

    plt.figure(1)
    plt.clf()
    plt.plot(fpr, tpr, label=f"ROC: {roc:.3f}")
    plt.legend()
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.xlabel("One-specificity")
    plt.ylabel("Sensitivity")
    plt.savefig(f"results/{name}/figs/{prefix}_roc.pdf")

    plt.figure(2)
    plt.clf()
    plt.plot(rs, ps, label=f"PR: {pr:.3f}")
    plt.legend()
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"results/{name}/figs/{prefix}_pr.pdf")


def draw_pca(nc: Array, mci: Array, name: str, prefix: str = "n", s: int = 10) -> None:
    """Draw the PCA."""
    os.makedirs(f"results/{name}/figs", exist_ok=True)
    arr = jnp.concatenate([nc, mci], axis=0)
    arr = arr.reshape(arr.shape[0], -1)
    pca1 = PCA(n_components=1).fit(arr)
    pca2 = PCA(n_components=2).fit(arr)
    pca3 = PCA(n_components=3).fit(arr)

    nc = nc.reshape(nc.shape[0], -1)
    mci = mci.reshape(mci.shape[0], -1)
    plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.scatter(pca1.transform(nc), np.zeros(nc.shape[0]), label="NC", c="green", s=s)
    plt.scatter(pca1.transform(mci), np.zeros(mci.shape[0]), label="MCI", c="red", s=s)
    plt.savefig(f"results/{name}/figs/{prefix}_pca1.pdf")

    plt.figure(2, figsize=(10, 10))
    plt.clf()
    plt.scatter(
        pca2.transform(nc)[:, 0], pca2.transform(nc)[:, 1], label="NC", c="green", s=s
    )
    plt.scatter(
        pca2.transform(mci)[:, 0], pca2.transform(mci)[:, 1], label="MCI", c="red", s=s
    )
    plt.savefig(f"results/{name}/figs/{prefix}_pca2.pdf")

    fig = plt.figure(3, figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        pca3.transform(nc)[:, 0],
        pca3.transform(nc)[:, 1],
        pca3.transform(nc)[:, 2],
        label="NC",
        c="green",
        s=s,
    )
    ax.scatter(
        pca3.transform(mci)[:, 0],
        pca3.transform(mci)[:, 1],
        pca3.transform(mci)[:, 2],
        label="MCI",
        c="red",
        s=s,
    )
    plt.savefig(f"results/{name}/figs/{prefix}_pca3.pdf")
