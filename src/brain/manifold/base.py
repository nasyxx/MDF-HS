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
filename : base.py
project  : brain
license  : GPL-3.0+

Base manifold.
"""
# Standard Library
from collections.abc import Callable

# Types
from jaxtyping import Array
from jax.typing import ArrayLike
from typing import NamedTuple

C1 = Callable[[ArrayLike], Array]
C2 = Callable[[Array, ArrayLike], Array]
C3 = Callable[[Array, Array, ArrayLike], Array]
C4 = Callable[[Array, Array, Array, ArrayLike], Array]


class Manifold(NamedTuple):
    """Manifold methods."""

    sqdist: C3
    inner: Callable[[Array, ArrayLike, Array, Array, bool], Array]
    proj: C2
    proj_tan: C3
    proj_tan0: C2
    expmap: C3
    expmap0: C2
    logmap: C3
    logmap0: C2
    mobius_add: C3
    mobius_matvec: C3
    mobius_scalar_mul: Callable[[ArrayLike, Array, ArrayLike], Array]
    ptransp: C4
    lambda_x: C2
    egrad2rgrad: C3
