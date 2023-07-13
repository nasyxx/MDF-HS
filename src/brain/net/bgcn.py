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
filename : bgcns.py
project  : brain
license  : GPL-3.0+

Brain GCN.

"""
# Standard Library
from collections.abc import Callable, Sequence
from functools import cache
from config import Net

# Types
from jaxtyping import Array

# JAX
import haiku as hk
import jax
import jax.numpy as jnp

# Local
from .utils import l2norm_without_sqrt, get_act
from brain.data.graph import Brain


class AF(hk.Module):
    """Functional Profile Learning."""

    def __init__(self, sigma: float = 2.0, name: str | None = None) -> None:
        """Init."""
        super().__init__(name)
        self.sigma = sigma

    def __call__(self, brain: Brain) -> Array:
        """Call."""
        t = brain.fmri.shape[-1]
        m = hk.get_parameter(
            "m",
            shape=(t, t),
            dtype=brain.fmri.dtype,
            init=hk.initializers.Identity(),
        )

        return jnp.exp(
            -l2norm_without_sqrt(
                jnp.einsum("mt,bnkt->bnkm", m, brain.fmri_pair), axis=-1
            )
            / (2 * (self.sigma**2))
        )


class GCN(hk.Module):
    """GCN layer."""

    def __init__(
        self,
        out: int,
        dropedge: bool = True,
        dropedge_rate: float = 0.8,
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ) -> None:
        """Initilize."""
        super().__init__(name)
        self.out = out
        self.dropedge = dropedge
        self.dropedge_rate = dropedge_rate
        self.l = hk.Linear(
            out,
            with_bias=with_bias,
            w_init=w_init or hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )

    def __call__(self, adj: Array, x: Array, train: bool = True) -> Array:
        """Call."""
        if self.dropedge and train:
            adj = hk.dropout(hk.next_rng_key(), self.dropedge_rate, adj)
        return adj @ self.l(x)


class BGCN(hk.Module):
    """Brain model."""

    def __init__(
        self,
        out_size: Sequence[int],
        dropedge: bool = True,
        dropout_rate: float = 0.6,
        dropedge_rate: float = 0.8,
        activation: Callable[[Array], Array] = jax.nn.relu,
        with_bias: bool = True,
        pearson: bool = True,
        name: str | None = None,
    ) -> None:
        """Init."""
        super().__init__(name=name)
        self.af = AF(name="FPL")
        self.dropout_rate = dropout_rate
        self.pears = pearson
        self.activation = activation
        self.bns = list(
            map(
                lambda _: hk.BatchNorm(
                    create_scale=True, create_offset=True, decay_rate=0.1, eps=1e-5
                ),
                out_size,
            )
        )
        self.gcns = list(
            map(
                lambda out: GCN(out, dropedge, dropedge_rate, with_bias),
                out_size,
            )
        )
        self.out = hk.Linear(2)

    def __call__(self, g: Brain, is_training: bool = False) -> Array:
        """Call."""
        beta1 = hk.get_parameter("beta1", [], init=jnp.ones)
        beta2 = hk.get_parameter("beta2", [], init=jnp.ones)

        adj_f = self.af(g)

        thetas = jnp.exp(-beta1) + jnp.exp(-beta2)
        theta1, theta2 = jnp.exp(-beta1) / thetas, jnp.exp(-beta2) / thetas

        adj = (
            jnp.eye(g.fmri.shape[-2], dtype=jnp.float32)
            + theta1 * g.adj_s
            + theta2 * adj_f
        )
        hk.set_state("adj", adj)

        x = g.fmri
        if self.pears:
            x = g.p

        for gcn, bn in zip(self.gcns, self.bns, strict=True):
            x = self.activation(bn(gcn(adj, x, is_training), is_training))
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        return self.out(x.reshape(x.shape[0], -1))


def build_model(net: Net) -> hk.TransformedWithState:
    """Build model."""

    def _model(g: Brain, train: bool = True) -> Array:
        return BGCN(
            net.out_size,
            net.dropedge,
            net.dropout_rate,
            net.dropedge_rate,
            activation=get_act(net.activation),
            pearson=net.pearson,
            name="model",
        )(g, train)

    return hk.transform_with_state(_model)
