"""Training loop + single-run driver for the Conditioning Barrier experiment.

Arms:
  diag        - vanilla unitary diagonal SSM (Adam, no preconditioning)   [Arm A, baseline]
  precond     - diag + per-transition Gram preconditioner (G+eps*lmax)^{-1/2}  [Arm B, the method]
  precond_nm  - precond + Frobenius-norm matching to the raw grad          [Arm B-norm ablation]
  pd          - PD-SSM ceiling (Adam)                                      [Arm C, solvability ceiling]

Every run logs: cross-entropy loss, in-distribution accuracy (T_train), OOD accuracy (T_ood),
and the RAW per-transition Gram condition number kappa(G) (diag-family only; the mechanistic proxy).
Seeds fix BOTH model init (mx key) and data generation (numpy rng) (confound Row 13).
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from groups import get_group, make_batch
from models import (
    diag_init, diag_forward, diag_forward_fast, pd_init, pd_forward, loss_fn, accuracy, TRANSITION_KEYS,
)
from precond import gram_and_kappa, precondition_transition_grads

ARMS = ("diag", "precond", "precond_nm", "pd")


def _make_model(arm, k, key, fast=False):
    if arm == "pd":
        H = 2 * k
        return pd_init(k, H, k, key), pd_forward
    d_c = k                       # state width 2k, matched across arms
    fwd = diag_forward_fast if fast else diag_forward   # fast = vectorized, machine-verified identical
    return diag_init(k, d_c, k, key), fwd


def eval_acc(forward, p, group, B, T, rng):
    x, y = make_batch(group, B, T, rng)
    return accuracy(forward, p, mx.array(x), mx.array(y)).item()


def run_experiment(group, arm, seed, steps, lr, batch=32, T_train=64, T_ood=256,
                   eps=1e-6, log_every=50, verbose=False, fast=False):
    assert arm in ARMS
    k = get_group(group)[1]
    p, forward = _make_model(arm, k, mx.random.key(seed), fast=fast)
    opt = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(seed)            # data rng (seeded together with init)
    eval_rng = np.random.default_rng(seed + 10_000)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(forward, pp, xx, yy))

    hist = {"step": [], "loss": [], "train_acc": [], "ood_acc": [], "kappa": []}
    is_diag = arm in ("diag", "precond", "precond_nm")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_batch(group, batch, T_train, rng)
        x, y = mx.array(x), mx.array(y)
        loss, grads = lg(p, x, y)

        kappa = float("nan")
        if is_diag and (step % log_every == 0 or step == 1):
            _, kappa, _, _ = gram_and_kappa(grads)   # RAW gram, diagnostic (confound Row 11)
        if arm == "precond":
            grads = precondition_transition_grads(grads, eps=eps, norm_match=False)
        elif arm == "precond_nm":
            grads = precondition_transition_grads(grads, eps=eps, norm_match=True)
        opt.update(p, grads)
        mx.eval(p, opt.state)

        if step % log_every == 0 or step == 1 or step == steps:
            ta = eval_acc(forward, p, group, 256, T_train, eval_rng)
            oa = eval_acc(forward, p, group, 256, T_ood, eval_rng)
            hist["step"].append(step)
            hist["loss"].append(float(loss.item()))
            hist["train_acc"].append(ta)
            hist["ood_acc"].append(oa)
            hist["kappa"].append(kappa)
            if verbose:
                print(f"  [{group}/{arm}/s{seed}] step {step:5d}  loss {loss.item():.4f}  "
                      f"acc {ta:.3f}  ood {oa:.3f}  kappa {kappa:.2e}")
    wall = time.time() - t0
    return {
        "group": group, "arm": arm, "seed": seed, "k": k, "steps": steps, "lr": lr,
        "batch": batch, "T_train": T_train, "T_ood": T_ood, "eps": eps,
        "final_train_acc": hist["train_acc"][-1], "final_ood_acc": hist["ood_acc"][-1],
        "final_loss": hist["loss"][-1], "wall_sec": wall, "history": hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T_train", type=int, default=64)
    ap.add_argument("--T_ood", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    res = run_experiment(a.group, a.arm, a.seed, a.steps, a.lr, a.batch,
                         a.T_train, a.T_ood, a.eps, a.log_every, a.verbose)
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "history"}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
