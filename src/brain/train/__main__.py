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
filename : __main__.py
project  : brain
license  : GPL-3.0+

Run the model
"""
# Utils
from collections.abc import Callable
from rich import print
import optax
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array

# Config
from config import config

# Local
from brain.data import Brain
from .base import main
from brain.net import build_bgcn, build_bhgcn


if __name__ == "__main__":
    conf = config()
    print(conf)
    if conf.model == "bgcn":
        model = build_bgcn(conf.net)
    elif conf.model == "bhgcn":
        model = build_bhgcn(conf)
    else:
        raise ValueError(f"Unknown model: {conf.model}")
    main(model, conf)
