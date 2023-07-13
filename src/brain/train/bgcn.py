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
date     : Feb 14, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : train_bgcns.py
project  : brain
license  : GPL-3.0+

Train the Brain GCN model.
"""

# Standard Library
from collections.abc import Callable, Iterator
from dataclasses import asdict
from functools import partial

# Types
from jaxtyping import Array

# Utils
from rich import print
from tqdm import tqdm

# JAX
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver

# Config
from config import Conf, config, Net

# Local
from brain.data import (
    Brain,
    Dataset,
    fold,
    read_all_brain,
    train_test_group_split,
)
from brain.data.utils import batchify
from brain.net.bgcn import build_model
from brain.utils import save_model


def loss_fn(
    labels: Array,
    model: hk.TransformedWithState,
    use_weight: bool = False,
) -> Callable[[hk.Params, hk.State, Array, Brain], tuple[Array, hk.State]]:
    """Loss function."""
    weight = sum(labels == 0) / sum(labels == 1)

    @jax.jit
    def _loss(
        params: hk.Params, states: hk.State, rng: Array, data: Brain
    ) -> tuple[jnp.ndarray, hk.State]:
        """Loss function."""
        logits, new_states = model.apply(params, states, rng, data, train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, data.label)
        if use_weight:
            loss = jnp.where(data.label == 0, loss, loss * weight)
        return loss.mean(), new_states

    return _loss


@jax.jit
def compare(pred: Array, label: Array) -> Array:
    """Compare the label and prediction."""
    lpred = jnp.argmax(jax.nn.softmax(pred), axis=1)
    return jnp.sum(lpred.squeeze() == label) / len(label)


def train(
    model: hk.TransformedWithState,
    fold_: int,
    train_data: list[Brain],
    val_data: list[Brain],
    test_data: list[Brain],
    conf: Conf,
) -> tuple[Array, Array]:
    """Train the model."""
    trgs, vlgs, tegs = (
        Dataset(train_data, conf),
        Dataset(val_data, conf),
        Dataset(test_data, conf),
    )
    train_graphs = next(trgs.loader(1000))
    test_graphs = next(tegs.loader())
    val_graphs = next(vlgs.loader())

    # Init the model
    rngs = hk.PRNGSequence(conf.seed)
    params, states = model.init(next(rngs), train_graphs, train=True)

    sche = optax.exponential_decay(
        conf.lr,
        transition_steps=50,
        decay_rate=0.99,
    )

    opt = optax.adamw(sche, weight_decay=1e-2)
    loss = loss_fn(train_graphs.label, model)
    solver = OptaxSolver(opt=opt, fun=loss, has_aux=True)
    opt_states = solver.init_state(params, states, next(rngs), test_graphs)
    update = jax.jit(solver.update)
    apply = jax.jit(partial(model.apply, train=False))
    # update = solver.update

    def steps(dataset: Dataset) -> Iterator[Brain]:
        """Generate Steps."""
        for _ in range(conf.epochs):
            yield from dataset.loader(conf.batch_size)

    losses = []
    acc = jnp.asarray(0.0)
    best_params = params
    best_states = states

    def _test() -> None:
        # import ipdb;ipdb.set_trace()
        tacc = compare(
            apply(params, states, next(rngs), train_graphs)[0],
            train_graphs.label,
        )
        vacc = compare(
            apply(params, states, next(rngs), val_graphs)[0],
            val_graphs.label,
        )
        nacc = compare(
            apply(params, states, None, test_graphs)[0],
            test_graphs.label,
        )
        bacc = compare(
            apply(best_params, best_states, None, test_graphs)[0],
            test_graphs.label,
        )
        bvacc = compare(
            apply(best_params, best_states, None, val_graphs)[0],
            val_graphs.label,
        )
        print(
            {
                "fold": fold_,
                "step": step,
                "loss": jnp.asarray(losses).mean().item(),
                "train_acc": tacc.item(),
                "val_acc": vacc.item(),
                "test_acc": nacc.item(),
                "best_acc": bacc.item(),
                "best_val_acc": bvacc.item(),
            }
        )
        losses.clear()

    for step, batch in enumerate(
        tqdm(steps(trgs), total=conf.epochs * len(list(trgs.loader(conf.batch_size))))
    ):
        params, opt_states = update(params, opt_states, states, next(rngs), batch)
        states = opt_states.aux
        losses.append(opt_states.value)
        new_acc = compare(
            apply(params, states, None, val_graphs)[0],
            val_graphs.label,
        )
        if new_acc > acc:
            acc = new_acc
            best_params = params
            best_states = states
            save_model(best_params, best_states, conf, name=f"best-{fold_}")

        if step % conf.chkt == 0:
            _test()

    bpred = apply(best_params, best_states, None, test_graphs)[0]
    acc = compare(bpred, test_graphs.label)
    pred = jnp.argmax(
        jax.nn.softmax(bpred),
        axis=1,
    )
    fpred = apply(params, states, None, test_graphs)[0]
    facc = compare(fpred, test_graphs.label)
    print(
        {
            "pred": pred,
            "label": test_graphs.label,
            "acc": acc,
        }
    )
    return acc, facc


if __name__ == "__main__":
    confi = config()
    print(asdict(confi))
