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
filename : bhgcn.py
project  : brain
license  : GPL-3.0+

Brain hyperbolic GCN.
"""
# Standard Library
from collections.abc import Callable, Sequence
from brain.data.graph import Brain
from config import Conf

# Types
from jaxtyping import Array
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import haiku as hk
from brain.manifold import poincare, Manifold
from .utils import get_act, l2norm_without_sqrt
from typing import Any


def init_wrap(
    manifold: Manifold, c: ArrayLike, f: hk.initializers.Initializer
) -> hk.initializers.Initializer:
    """Init wrap."""

    def _wrap(*args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        return manifold.expmap0(f(*args, **kwargs), c)

    return _wrap


class HAct(hk.Module):
    """Hyperbolic activation function."""

    def __init__(
        self,
        manifold: Manifold,
        c_in: float = 1.0,
        c_out: float = 1.0,
        act: Callable[[Array], Array] = jax.nn.leaky_relu,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def __call__(self, x: Array, tc: ArrayLike = 1.0) -> Array:
        """Call."""
        manifold = self.manifold
        cin = jnp.asarray(self.c_in, dtype=x.dtype) * tc
        cout = jnp.asarray(self.c_out, dtype=x.dtype) * tc
        return manifold.proj(
            manifold.expmap0(
                manifold.proj_tan0(self.act(manifold.logmap0(x, cin)), cout), cout
            ),
            cout,
        )


class HLinear(hk.Module):
    """Hyperbolic linear layer."""

    def __init__(
        self,
        output_size: int,
        manifold: Manifold = poincare,
        c: float = 1.0,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        with_bias: bool = True,
        hinput: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__(name=name)
        self.output_size = output_size
        self.manifold = manifold
        self.c = c
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.with_bias = with_bias
        self.hinput = hinput

    def __call__(self, x: Array, tc: ArrayLike = 1.0) -> Array:
        """Forward."""
        input_size = x.shape[-1]
        dtype = x.dtype
        c = self.c * tc

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / jnp.sqrt(input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = hk.get_parameter("w", [input_size, self.output_size], dtype, init=w_init)

        if self.hinput:
            out = self.manifold.mobius_matvec(x, w, c)
        else:
            out = jnp.matmul(x, w)
            out = self.manifold.expmap0(out, c)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            out = self.manifold.mobius_add(out, b, c)

        return self.manifold.proj(out, c)


class HAF(hk.Module):
    """Hyperbolic Functional Profile Learning."""

    def __init__(
        self,
        manifold: Manifold,
        c: float = 1.0,
        sigma: float = 2.0,
        name: str | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(name=name)
        self.c = c
        self.sigma = sigma
        self.manifold = manifold

    def __call__(self, fmri_pair: Array, tc: ArrayLike = 1.0) -> Array:
        """Call."""
        t = fmri_pair.shape[-1]
        m = hk.get_parameter(
            "m", shape=(t, t), dtype=fmri_pair.dtype, init=hk.initializers.Identity()
        )
        c = self.c * tc
        return self.manifold.proj(
            self.manifold.expmap0(
                jnp.exp(
                    -l2norm_without_sqrt(
                        jnp.einsum("mt,bnkt->bnkm", m, fmri_pair), axis=-1
                    )
                    / (2 * (self.sigma**2))
                ),
                c,
            ),
            c,
        )


class HGCN(hk.Module):
    """Hyperbolic GCN."""

    def __init__(  # noqa: PLR0913
        self,
        out: int,
        manifold: Manifold,
        c: float = 1.0,
        dropout_rate: float = 0.6,
        dropedge: bool = True,
        dropedge_rate: float = 0.8,
        with_bias: bool = True,
        hinput: bool = True,
        use_att: bool = True,
        heads: int = 4,
        concat: bool = False,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the module."""
        super().__init__(name=name)
        self.out = out
        self.manifold = manifold
        self.c = c
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.dropout_rate = dropout_rate
        self.dropedge = dropedge
        self.dropedge_rate = dropedge_rate
        self.hinput = hinput
        self.heads = heads
        self.use_att = use_att
        self.concat = concat
        if use_att:
            out = out * heads
        self.l = HLinear(
            out,
            manifold=manifold,
            c=c,
            w_init=w_init,
            b_init=b_init,
            with_bias=with_bias,
            hinput=hinput,
        )

    def __call__(
        self,
        adj: Array,
        x: Array,
        tc: ArrayLike = 1.0,
        train: bool = True,
    ) -> Array:
        """Forward."""
        c = self.c * tc

        if self.dropedge and train:
            adj = hk.dropout(hk.next_rng_key(), self.dropedge_rate, adj)

        if self.use_att:
            # b x (148 x 148)
            batch_size, n_n, n_f = x.shape

            w = hk.get_parameter(
                "w",
                shape=(n_f, self.heads * self.out),
                dtype=x.dtype,
                init=self.w_init
                or hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            )

            xw = self.manifold.proj(
                self.manifold.expmap0(self.manifold.logmap0(x, c) @ w, c), c
            )
            # xw = jax.vmap(lambda xi: self.manifold.mobius_matvec(xi, w, c))(x)

            if self.with_bias:
                b = hk.get_parameter(
                    "b",
                    shape=(self.heads * self.out,),
                    dtype=x.dtype,
                    init=self.b_init,
                )
                xw = self.manifold.mobius_add(xw, b, c)

            xw = jnp.reshape(xw, (batch_size, n_n, self.heads, -1))

            alpha = hk.get_parameter(
                "alpha",
                shape=(2 * self.out,),
                dtype=x.dtype,
                init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
            )

            xw_t = self.manifold.logmap0(xw, c)

            logit_p = (xw_t * alpha[..., : alpha.shape[0] // 2]).sum(axis=-1)
            logit_c = (xw_t * alpha[..., alpha.shape[0] // 2 :]).sum(axis=-1)
            attn = logit_p[..., None, :] + logit_c[:, None, ...]
            attn = jax.nn.leaky_relu(attn, 0.2)

            zero_vec = -1e12 * jnp.ones_like(attn)
            mask = adj[..., None]
            attn = jnp.where(mask > 0, attn, zero_vec)

            attn = jax.nn.softmax(attn, axis=-1)

            if self.dropout_rate and train:
                attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn)

            xw = jnp.einsum("bijh,bjhc->bihc", attn, xw_t)

            if self.concat:
                xw = jnp.reshape(xw, (batch_size, n_n, -1))
            else:
                xw = xw.mean(axis=2)

            return self.manifold.proj(self.manifold.expmap0(xw, 0), c)

        x = self.l(x)
        if self.with_bias:
            b = hk.get_parameter("b", [self.out], dtype=x.dtype, init=self.b_init)
            x = self.manifold.mobius_add(x, b, c)
        return self.manifold.proj(x, c)


class BHGCN(hk.Module):
    """Brain hyperbolic GCN."""

    def __init__(  # noqa: PLR0913
        self,
        out_size: Sequence[int],
        manifold: str = "poincare",
        c: float = 1.0,
        dropedge: bool = True,
        dropout_rate: float = 0.6,
        dropedge_rate: float = 0.8,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        with_bias: bool = True,
        use_att: bool = True,
        heads: int = 4,
        concat: bool = False,
        hinput: bool = True,
        pearson: bool = True,
        trainc: bool = False,
        name: str | None = None,
    ) -> None:
        """Init."""
        super().__init__(name=name)
        self.manifold = {"poincare": poincare}[manifold]
        self.dropout_rate = dropout_rate
        self.pears = pearson
        self.c = c
        self.trainc = trainc

        w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.bns = list(
            map(
                lambda _: hk.BatchNorm(
                    create_scale=True, create_offset=True, decay_rate=0.1, eps=1e-5
                ),
                out_size,
            )
        )
        self.hgcns = list(
            map(
                lambda out: HGCN(
                    out,
                    self.manifold,
                    c,
                    dropout_rate,
                    dropedge,
                    dropedge_rate,
                    with_bias,
                    hinput=hinput,
                    use_att=use_att,
                    heads=heads,
                    concat=concat,
                    w_init=w_init,
                ),
                out_size,
            )
        )
        self.af = HAF(self.manifold)
        self.act = HAct(self.manifold, c, c, activation)
        self.out = HLinear(2, self.manifold, c, hinput=False)

    def __call__(self, g: Brain, is_training: bool = True) -> Array:
        """Forward."""
        beta1 = hk.get_parameter("beta1", [], init=jnp.ones)
        beta2 = hk.get_parameter("beta2", [], init=jnp.ones)
        c = jnp.asarray(self.c, dtype=g.fmri.dtype)
        tc: ArrayLike = 1.0
        if self.trainc:
            tc = hk.get_parameter("tc", [], init=jnp.ones)
        c = c * tc

        adj_f = self.af(g.fmri_pair, tc)

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

        for hgcn, bn in zip(self.hgcns, self.bns, strict=True):
            x = hgcn(adj, x, tc, train=is_training)
            x = self.manifold.proj(
                self.manifold.expmap0(
                    bn(self.manifold.logmap0(x, c), is_training=is_training),
                    c,
                ),
                c,
            )
            x = self.act(x, tc)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x = self.manifold.logmap0(x.reshape(x.shape[0], -1), c)
        return self.out(x, tc)


def build_model(conf: Conf) -> hk.TransformedWithState:
    """Build model."""
    net = conf.net

    def _model(g: Brain, train: bool = True) -> Array:
        return BHGCN(
            net.out_size,
            "poincare",
            conf.c,
            net.dropedge,
            net.dropout_rate,
            net.dropedge_rate,
            activation=get_act(net.activation),
            use_att=net.use_att,
            heads=net.heads,
            concat=net.concat,
            pearson=net.pearson,
            trainc=net.trainc,
            name="model",
        )(g, is_training=train)

    return hk.transform_with_state(_model)
