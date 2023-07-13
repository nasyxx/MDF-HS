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
from collections.abc import Callable, Iterator
import json
from brain.data.input import read_all_brain
from brain.data.utils import batchify, fold, train_test_group_split
from brain.manifold import poincare
from brain.manifold.base import Manifold
from brain.optim import scale_by_radam, apply_updates, scale_by_rsgd, apply_rsgd_updates
from optax._src.alias import _scale_by_learning_rate  # type: ignore[attr-defined]
from .metric import auc_fn, diff_adj, draw_auc, draw_heat, prf_fn, draw_pca
import haiku as hk
from rich import print

# Types
from jaxtyping import Array
import optax

# JAX
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Config
from config import Conf
from brain.data import Brain, Dataset
from functools import partial
from typing import TypedDict
from brain.utils import save_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Result(TypedDict):
    """Result."""

    name: str
    params: hk.Params
    states: hk.State
    pred: Array
    label: Array
    acc: Array
    roc: Array
    pr: Array
    precision: Array
    recall: Array
    f1: Array
    spe: Array
    sen: Array


def loss_fn(
    labels: Array,
    model: hk.TransformedWithState,
    manifold: Manifold,
    c: ArrayLike = 1.0,
    use_manifold: bool = False,
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
        # if use_manifold:
        #     logits = manifold.logmap0(logits, c)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, data.label)
        if use_weight:
            loss = jnp.where(data.label == 0, loss, loss * weight)
        return loss.mean(), new_states

    return _loss


@jax.jit
def compare(pred: Array, label: Array) -> Array:
    """Compare the label and prediction."""
    lpred = jnp.argmax(jax.nn.softmax(pred), axis=1)
    return jnp.mean(lpred.squeeze() == label)


def gen_result(
    model: hk.TransformedWithState,
    params: hk.Params,
    states: hk.State,
    brain: Brain,
    name: str,
) -> Result:
    """Generate result."""
    pred, _ = model.apply(params, states, None, brain, train=False)
    lpred = jnp.argmax(jax.nn.softmax(pred), axis=1)
    acc = compare(pred, brain.label)
    _, _, roc, pr = auc_fn(brain.label, lpred)
    p, r, f = prf_fn(brain.label, lpred)

    tn, fp, fn, tp = confusion_matrix(brain.label, lpred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)

    return Result(
        name=name,
        params=params,
        states=states,
        pred=lpred,
        label=brain.label,
        acc=acc,
        roc=roc,
        pr=pr,
        precision=p,
        recall=r,
        f1=f,
        spe=specificity,
        sen=sensitivity,
    )


def train(  # noqa: PLR0915
    model: hk.TransformedWithState,
    manifold: Manifold,
    fold_: int,
    train_data: list[Brain],
    val_data: list[Brain],
    test_data: list[Brain],
    conf: Conf,
) -> tuple[Result, Result, Result, Result]:
    """Train the model."""
    conf.log(f"Fold {fold_}")
    trgs, vlgs, tegs = (
        Dataset(train_data, conf),
        Dataset(val_data, conf),
        Dataset(test_data, conf),
    )
    train_graphs = next(trgs.loader(1000))
    test_graphs = next(tegs.loader(conf.bt and 1000 or None))
    val_graphs = next(vlgs.loader())

    # Init the model
    rngs = hk.PRNGSequence(conf.seed)
    params, states = model.init(next(rngs), train_graphs, train=True)

    sche = optax.exponential_decay(
        conf.lr,
        transition_steps=50,
        decay_rate=0.95,  # conf.optim == "rsgd" and 0.95 or 0.99,
    )

    def get_opt(name: str) -> optax.GradientTransformation:
        """Get the optimizer."""
        if name == "adam":
            return optax.adamw(sche, weight_decay=1e-2)
        if name == "radam":
            return optax.chain(
                scale_by_radam(poincare, c=conf.c, weight_decay=1e-2),
                _scale_by_learning_rate(sche),
            )
        if name == "rsgd":
            return optax.chain(
                scale_by_rsgd(poincare, c=conf.c),
                _scale_by_learning_rate(sche),
            )
        raise ValueError(f"Unknown optimizer: {name}")

    opt = get_opt(conf.optim)
    loss = loss_fn(train_graphs.label, model, manifold, conf.model == "bhgcn")
    opt_states = opt.init(params)
    # solver = OptaxSolver(opt=opt, fun=loss, has_aux=True)
    # opt_states = solver.init_state(params, states, next(rngs), test_graphs)
    # update = jax.jit(solver.update)

    @jax.jit
    def update(params, opt_states, states, rng, batch):  # type: ignore[no-untyped-def] # noqa: ANN
        """Update the model."""
        (value, new_states), grads = jax.value_and_grad(loss, has_aux=True)(
            params, states, rng, batch
        )
        updates, new_opt_states = opt.update(grads, opt_states, params)
        if conf.optim == "adam":
            params = optax.apply_updates(params, updates)
        if conf.optim == "radam":
            params = apply_updates(poincare, conf.c, params, updates)
        if conf.optim == "rsgd":
            params = apply_rsgd_updates(poincare, conf.c, params, updates)
        return params, new_opt_states, new_states, value

    # apply = partial(model.apply, train=False)
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
        nacc = compare(
            apply(params, states, None, test_graphs)[0],
            test_graphs.label,
        )
        bacc = compare(
            apply(best_params, best_states, None, test_graphs)[0],
            test_graphs.label,
        )
        rr = {
            "fold": fold_,
            "step": step,
            "loss": jnp.asarray(losses).mean().item(),
            "train_acc": tacc.item(),
            "test_acc": nacc.item(),
            "best_test_acc": bacc.item(),
        }
        if conf.val:
            vacc = compare(
                apply(params, states, next(rngs), val_graphs)[0],
                val_graphs.label,
            )
            bvacc = compare(
                apply(best_params, best_states, None, val_graphs)[0],
                val_graphs.label,
            )
            rr["val_acc"] = vacc.item()
            rr["best_val_acc"] = bvacc.item()
        conf.log(rr)
        losses.clear()

    for step, batch in enumerate(
        tq := tqdm(
            steps(trgs),
            total=conf.epochs * len(list(trgs.loader(conf.batch_size))),
            ncols=120,
            desc=f"Fold {fold_}, Loss nn, Acc nn, LR nn",
        )
    ):
        params, opt_states, states, value = update(
            params, opt_states, states, next(rngs), batch
        )
        losses.append(value)
        new_acc = compare(
            apply(params, states, None, test_graphs)[0],
            test_graphs.label,
        )
        lr = ""
        if conf.optim == "rsgd":
            lr = f", LR {sche(opt_states[1].count).item():.4f}"  # type: ignore[union-attr,index]  # noqa: E501
        tq.set_description(
            f"Fold {fold_}, Loss {value:.4f}, Acc {new_acc:.4f}{lr}",
        )
        if new_acc > acc:
            acc = new_acc
            best_params = params
            best_states = states
            save_model(
                best_params, best_states, conf, name=conf.name, prefix=f"best_{fold_}"
            )

        if step % conf.chkt == 0:
            _test()

    save_model(params, states, conf, name=conf.name, prefix=f"final_{fold_}")
    _test()

    res1 = gen_result(model, params, states, test_graphs, "final_test")
    res2 = gen_result(model, best_params, best_states, test_graphs, "best_test")
    if conf.val:
        all_graphs = jax.tree_map(
            lambda *x: jnp.concatenate(x), train_graphs, val_graphs, test_graphs
        )
    else:
        all_graphs = jax.tree_map(
            lambda *x: jnp.concatenate(x), train_graphs, test_graphs
        )
    res3 = gen_result(model, params, states, all_graphs, "final_all")
    res4 = gen_result(model, params, states, all_graphs, "best_all")

    return res1, res2, res3, res4


def resr(fold_: int, res: tuple[Result, ...]) -> dict[str, int | dict[str, float]]:
    """Print the result."""
    return {
        "fold": fold_,
        "acc": {
            "final_test": res[0]["acc"].item(),
            "best_test": res[1]["acc"].item(),
            "final_all": res[2]["acc"].item(),
            "best_all": res[3]["acc"].item(),
        },
        "roc": {
            "final_test": res[0]["roc"].item(),
            "best_test": res[1]["roc"].item(),
            "final_all": res[2]["roc"].item(),
            "best_all": res[3]["roc"].item(),
        },
        "pr": {
            "final_test": res[0]["pr"].item(),
            "best_test": res[1]["pr"].item(),
            "final_all": res[2]["pr"].item(),
            "best_all": res[3]["pr"].item(),
        },
        "precision": {
            "final_test": res[0]["precision"].item(),
            "best_test": res[1]["precision"].item(),
            "final_all": res[2]["precision"].item(),
            "best_all": res[3]["precision"].item(),
        },
        "recall": {
            "final_test": res[0]["recall"].item(),
            "best_test": res[1]["recall"].item(),
            "final_all": res[2]["recall"].item(),
            "best_all": res[3]["recall"].item(),
        },
        "f1": {
            "final_test": res[0]["f1"].item(),
            "best_test": res[1]["f1"].item(),
            "final_all": res[2]["f1"].item(),
            "best_all": res[3]["f1"].item(),
        },
    }


def main(model: hk.TransformedWithState, conf: Conf) -> None:  # noqa: PLR0915
    """Run the main function."""
    conf.log("Loading data")
    data = read_all_brain(conf)
    print(
        hk.experimental.tabulate(
            model, columns=("module", "owned_params", "params_size", "params_bytes")
        )(batchify(data[:2]))
    )
    ind, ted = train_test_group_split(data, conf.k, conf.seed, conf.group)
    res: dict[int, tuple[Result, Result, Result, Result]] = {}
    manifold = poincare
    if not conf.val:
        ind = data
    for i, (train_data, val_data) in enumerate(
        fold(ind, conf.k, conf.seed, conf.group)
    ):
        conf.log(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(ted)}")
        if conf.val:
            res[i] = train(model, manifold, i, train_data, val_data, ted, conf)
        else:
            res[i] = train(model, manifold, i, train_data, val_data, val_data, conf)

    conf.log("Evaluation")

    with open(f"results/{conf.name}/res.json", "w") as f:
        outs: dict = {}
        outs["summary"] = {}
        for me in ("acc", "roc", "pr", "precision", "recall", "f1", "spe", "sen"):
            tmean = jnp.stack(list(map(lambda v: v[1][me], res.values()))).mean().item()
            ftmean = (
                jnp.stack(list(map(lambda v: v[0][me], res.values()))).mean().item()
            )
            amean = jnp.stack(list(map(lambda v: v[3][me], res.values()))).mean().item()
            famean = (
                jnp.stack(list(map(lambda v: v[2][me], res.values()))).mean().item()
            )

            tbest = max(res.values(), key=lambda v: v[1][me].item())[1][me].item()
            fbest = max(res.values(), key=lambda v: v[0][me].item())[0][me].item()
            abest = max(res.values(), key=lambda v: v[3][me].item())[3][me].item()
            fabest = max(res.values(), key=lambda v: v[2][me].item())[2][me].item()
            outs["summary"][me] = {
                "tbest": tbest,
                "fbest": fbest,
                "abest": abest,
                "fabest": fabest,
                "tmean": tmean,
                "ftmean": ftmean,
                "amean": amean,
                "famean": famean,
            }
        outs["fold"] = []
        for i, r in res.items():
            conf.log(resr(i, r))
            outs["fold"].append(resr(i, r))
        f.write(json.dumps(outs, indent=2))
        f.write("\n")

    print(
        "Best Test Mean:",
        outs["summary"]["acc"]["tmean"],
        "Final Test Mean:",
        outs["summary"]["acc"]["ftmean"],
        "Best All Mean:",
        outs["summary"]["acc"]["abest"],
        "Final All Mean:",
        outs["summary"]["acc"]["famean"],
        sep="\n",
    )

    # all test
    labels = jnp.concatenate(list(map(lambda v: v[1]["label"], res.values())))
    preds = jnp.concatenate(list(map(lambda v: v[1]["pred"], res.values())))
    # _, _, roc, pr = auc_fn(labels, preds)
    draw_auc(labels, preds, conf.name, "test")

    # all
    labels = jnp.concatenate(list(map(lambda v: v[3]["label"], res.values())))
    preds = jnp.concatenate(list(map(lambda v: v[3]["pred"], res.values())))
    # _, _, roc, pr = auc_fn(labels, preds)
    draw_auc(labels, preds, conf.name, "all")

    nc = list(filter(lambda x: x.label == 0, data))
    mci = list(filter(lambda x: x.label == 1, data))
    bnc = batchify(nc)
    bmci = batchify(mci)

    max_res = max(res.values(), key=lambda x: x[3]["acc"].item())[3]
    ncp, ncstates = model.apply(
        max_res["params"], max_res["states"], None, batchify(nc), train=False
    )
    mcip, mcistates = model.apply(
        max_res["params"], max_res["states"], None, batchify(mci), train=False
    )
    dpred, _ = model.apply(
        max_res["params"], max_res["states"], None, batchify(data), train=False
    )
    dlpred = jnp.argmax(jax.nn.softmax(dpred), axis=-1)
    dncp = dpred[dlpred == 0]
    dmcip = dpred[dlpred == 1]

    # adj
    draw_pca(bnc.adj_s, bmci.adj_s, conf.name, "before")
    draw_pca(ncstates["model"]["adj"], mcistates["model"]["adj"], conf.name, "after")

    # heat
    ncbase = jnp.mean(bnc.adj_s, axis=0)
    mcibase = jnp.mean(bmci.adj_s, axis=0)
    draw_heat(ncbase, conf.name, "ncbase", 0, 5)
    draw_heat(mcibase, conf.name, "mcibase", 0, 5)

    plt.figure(1)
    plt.clf()
    plt.scatter(ncp[:, 0], ncp[:, 1], c="green", label="NC", s=3)
    plt.scatter(mcip[:, 0], mcip[:, 1], c="red", label="MCI", s=3)
    plt.legend()
    plt.savefig(f"results/{conf.name}/figs/ncmci.pdf")

    plt.figure(2)
    plt.clf()
    plt.scatter(dncp[:, 0], dncp[:, 1], c="green", label="NC", s=3)
    plt.scatter(dmcip[:, 0], dmcip[:, 1], c="red", label="MCI", s=3)
    plt.legend()
    plt.savefig(f"results/{conf.name}/figs/ncmci_pred.pdf")

    ncadjs = ncstates["model"]["adj"].mean(axis=0)
    mciadjs = mcistates["model"]["adj"].mean(axis=0)
    draw_heat(ncadjs, conf.name, "ncbest", 0, 5)
    draw_heat(mciadjs, conf.name, "mcibest", 0, 5)

    ncdecrease = jnp.minimum(ncadjs - ncbase, 0)
    ncincrease = jnp.maximum(ncadjs - ncbase, 0)
    mcidecrease = jnp.minimum(mciadjs - mcibase, 0)
    mciincrease = jnp.maximum(mciadjs - mcibase, 0)

    draw_heat(ncincrease, conf.name, "ncincrease", 0, 2)
    draw_heat(ncdecrease, conf.name, "ncdecrease", -2, 0)
    draw_heat(mciincrease, conf.name, "mciincrease", 0, 2)
    draw_heat(mcidecrease, conf.name, "mcidecrease", -2, 0)
