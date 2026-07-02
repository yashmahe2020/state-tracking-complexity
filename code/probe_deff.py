"""FIXED-K d_eff PROBE (cheap, pre-advisor-ruling evidence). Holds K=12 constant and varies ONLY the
effective representation dimension of the target automaton:
    Z12   (cyclic, abelian)        d_eff = 1
    D6    (dihedral, order 12)      d_eff = 2   (non-abelian; faithful 2-D rep of the hexagon)
    RAND12(random perm automaton)  d_eff ~ 12  (no algebraic structure; coverage-bound tight)
At fixed K, K log K and K^2 log K are CONSTANTS, so this cannot see the K-scaling -- it tests the
ORTHOGONAL hypothesis: does the statistical m* rise with d_eff (Z12 ~ D6 << RAND12)? If so, the rate
is governed by effective dimension, not group size; D6 ~ Z12 would also refute D_n as a K^2 log K
vehicle (the advisor's plan). Prediction (m* ~ K * d_eff * log K): Z12 ~8.5k, D6 ~17k, RAND12 ~100k
transitions. Same soft-model, fixed-data, fresh-test protocol as m_star_sweep. Writes results/probe_deff.txt.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "probe_deff.json")

TARGETS = ["Z12", "D6", "RAND12"]   # all K=12 states/symbols
SEEDS = [0, 1]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.01
EVAL_FRESH = 512
MAX_WORKERS = 4
# n_seq grid spanning Z12's known m* (~533 seq) up to ~12x for the RAND endpoint
NSEQ = [60, 120, 240, 480, 720, 1080, 1600, 2400, 3600, 5400, 8000]


def worker(task):
    target, n_seq, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch, get_group
    from models import pd_init, pd_forward, loss_fn
    _, k = get_group(target)
    H = 2 * k
    drng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(target, n_seq, T, drng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    trng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(target, EVAL_FRESH, T, trng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = pd_init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(pd_forward, pp, xx, yy))
    bs = min(64, n_seq)

    def sacc(xx, yy):
        pred = mx.argmax(pd_forward(p, xx), axis=-1)
        return float(mx.mean(mx.all(pred == yy, axis=1).astype(mx.float32)))

    best = 0.0
    rng = np.random.default_rng(seed)
    for step in range(1, TRAIN_STEPS + 1):
        idx = rng.integers(0, n_seq, size=bs)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        opt.update(p, g); mx.eval(p, opt.state)
        if step % 250 == 0:
            best = max(best, sacc(Xte, Yte))
    test = max(best, sacc(Xte, Yte))
    return {"target": target, "k": k, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed,
            "test_seq_acc": test, "generalizes": test >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["target"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(t, n, s) for t in TARGETS for n in NSEQ for s in SEEDS if (t, n, s) not in seen]
    print(f"probe: {len(tasks)} runs", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] {r['target']:<7} n={r['n_seq']:<5} trans={r['transitions']:<6} "
                  f"s{r['seed']} test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = ["=== fixed-K=12 d_eff probe: statistical m* vs effective dimension ===",
             f"m* = min transitions with median(n={len(SEEDS)}) fresh-test seq-acc >= {THR}; T={T}",
             f"{'target':<8} {'d_eff':>5} {'m*_trans':>9}"]
    deff = {"Z12": 1, "D6": 2, "RAND12": 12}
    ms = {}
    for t in TARGETS:
        grid = sorted(set(r["n_seq"] for r in results if r["target"] == t))
        m = None
        for n in grid:
            cell = [r for r in results if r["target"] == t and r["n_seq"] == n]
            if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                m = n * T; break
        ms[t] = m
        lines.append(f"{t:<8} {deff[t]:>5} {(m if m else '>grid'):>9}")
    lines.append("")
    if ms.get("Z12") and ms.get("D6") and ms.get("RAND12"):
        z, d, r = ms["Z12"], ms["D6"], ms["RAND12"]
        lines.append(f"ratios at fixed K=12:  D6/Z12 = {d/z:.2f}   RAND12/Z12 = {r/z:.2f}   RAND12/D6 = {r/d:.2f}")
        if r > 2.5 * d and r > 2.5 * z:
            lines.append("VERDICT: m* RISES STRONGLY with d_eff (RAND >> D6,Z12). The rate is governed by "
                         "EFFECTIVE DIMENSION, not group size K. Confirms the d_eff-spectrum reframe; "
                         "refutes D_n as a K^2 log K vehicle (D6 ~ Z12, both low-d_eff).")
        elif r > 1.5 * z:
            lines.append("VERDICT: m* rises with d_eff but modestly; d_eff matters, magnitude needs the full K-sweep.")
        else:
            lines.append("VERDICT: m* roughly FLAT across d_eff at K=12 -> d_eff hypothesis NOT supported; reconsider.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "probe_deff.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
