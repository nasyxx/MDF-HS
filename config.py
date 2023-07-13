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
filename : config.py
project  : brain
license  : GPL-3.0+

Configuration for brain project.
"""
# Standard Library
from dataclasses import asdict, dataclass, field

# Types
from typing import Annotated, cast

# Utils
from rich import console, print, pretty

# Config
from smile_config import from_dataclass


@dataclass
class Net:
    """Parameters for brain."""

    out_size: tuple[int, ...] = field(default_factory=lambda: (148, 296))
    activation: str = "leaky_relu"
    leaky_relu_slope: float = 0.2
    dropedge: bool = True
    dropedge_rate: float = 0.8
    dropout: bool = True
    dropout_rate: float = 0.6
    use_att: bool = False
    heads: int = 4
    concat: bool = False
    pearson: bool = True
    trainc: bool = False


@dataclass
class Conf:
    """Configuration for brain."""

    data_path: str = "data/Data_MIA/Data_MIA_deepFusion"
    model_dir: str = "results"

    seed: int = 7
    reload: bool = False
    norm: bool = True
    val: bool = True
    split: bool = True
    group: bool = True

    batch_size: int = 1000
    epochs: int = 200
    chkt: int = 50
    lr: float = 0.001
    k: int = 5
    optim: str = "adam"

    model: str = "bgcn"
    manifold: Annotated[str, "(e)uclidean|(p)oincare"] = "e"
    c: float = 1.0
    bt: bool = False

    net: Net = field(default_factory=Net)

    def __post_init__(self) -> None:
        """Post init."""
        self.manifold = (
            (self.manifold.startswith("p") or self.model == "bhgcn") and "poincare"
        ) or "euclidean"
        self.console = console.Console()
        if self.net.trainc:
            self.warn("net.trainc")
        if self.optim == "radam":
            self.warn("optim.radam")

    @property
    def name(self) -> str:
        """Generate name of the configuration."""
        model, manifold, c = self.model, self.manifold, self.c
        seed, norm, split, group, val = (
            self.seed,
            self.norm,
            self.split,
            self.group,
            self.val,
        )
        lr, k, optim, bt = self.lr, self.k, self.optim, self.bt
        out_size, activation = self.net.out_size, self.net.activation
        dropedge = self.net.dropedge
        use_att, heads, concat = (self.net.use_att, self.net.heads, self.net.concat)
        trainc = self.net.trainc
        return (
            f"{model=}.{manifold=}.{c=}."
            f"{seed=}.{norm=}.{split=}.{group=}.{val=}.{bt=}."
            f"{dropedge=}.{lr=}.{k=}.{optim=}."
            f"{out_size=}.{activation=}.{trainc=}."
            f"{use_att=}.{heads=}.{concat=}."
        )

    def warn(self, name: str) -> None:
        """Warn."""
        self.console.log(f"[bold red]Warning:[/bold red] {name} is currently broken.")

    def log(self, msg: str | object) -> None:
        """Log message."""
        if not isinstance(msg, str):
            msg = pretty.Pretty(msg)
        self.console.log(msg, justify="left")


def config() -> Conf:
    """Get configuration."""
    return cast(Conf, from_dataclass(Conf()).config)


if __name__ == "__main__":
    print(asdict(config()))
