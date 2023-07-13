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
date     : Feb 24, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : brain
license  : GPL-3.0+

Utils
"""
import pickle
from config import Conf
import haiku as hk
import os
from pathlib import Path


def save_model(
    params: hk.Params, states: hk.State, conf: Conf, name: str, prefix: str = "best"
) -> None:
    """Save model."""
    os.makedirs(path := f"{conf.model_dir}/{name}", exist_ok=True)
    with open(f"{path}/{prefix}.pkl", "wb") as f:
        pickle.dump((params, states), f)


def load_model(conf: Conf, name: str = "best") -> tuple[hk.Params, hk.State]:
    """Load model."""
    path = sorted(Path(conf.model_dir).glob(f"{name}.*.pkl"))[-1]
    with path.open("rb") as f:
        return pickle.load(f)
