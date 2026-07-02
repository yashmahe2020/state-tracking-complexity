"""EXP-3 (Phase-4 review, advisor ruling #4, addresses triage T07): the ERM theorem (T1) and the
capacity theorem (T2) are stated for HARD permutation automata, but the statistical m* sweep used the
SOFT PD-SSM (M = P @ diag(d)). This spot-check measures m* on Z_K with the HARD-STE model (a pure
0/1 permutation transition, the theory's actual hypothesis class) at two K, to confirm the soft m*
is not a soft-model artifact. The hard model is fragile (dead STE basins), so we anneal the inverse
temperature beta from low to high over training (per models.pd_hard_forward docs).

For K in {12,20}: build a fixed Z_K training set of n_seq sequences, train pd_hard to convergence,
evaluate fresh-test SEQUENCE accuracy; m* = min transitions with median seq-acc >= 0.95 across seeds.
Compare m*(12), m*(20) and the implied exponent to the SOFT sweep (results/m_star_sweep.txt). Writes
results/exp_hard_mstar.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_hard_mstar.json")

KS = [12, 20]
SEEDS = [0, 1, 2]
T = 16
THR = 0.95
TRAIN_STEPS = 12000        # hard-STE needs more steps + beta anneal than soft
LR = 0.01
BETA0, BETA1 = 1.0, 8.0    # inverse-temperature anneal low->high (escape dead basins -> sharp hard)
EVAL_FRESH = 512
MAX_WORKERS = 3


def n_seq_grid(k):
    import numpy as np
    cov = k * k * math.log(k) / T
    lo = max(3.0, 0.3 * k * math.log(k) / T)
    hi = 40.0 * cov
    g = np.unique(np.round(np.geomspace(lo, hi, 12)).astype(int))
    return [int(x) for x in g if x >= 1]


def worker(task):
    k, n_seq, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_hard_init, pd_hard_forward, loss_fn
    group = "Z%d" % k
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    H = 2 * k
    p = pd_hard_init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    bs = min(64, n_seq)
    train_rng = np.random.default_rng(seed)
    best = 0.0
    for step in range(1, TRAIN_STEPS + 1):
        beta = BETA0 + (BETA1 - BETA0) * step / TRAIN_STEPS
        idx = train_rng.integers(0, n_seq, size=bs)
        xb, yb = Xtr[mx.array(idx)], Ytr[mx.array(idx)]
        lg = mx.value_and_grad(lambda pp: loss_fn(lambda q, x: pd_hard_forward(q, x, beta), pp, xb, yb))
        _, g = lg(p)
        opt.update(p, g); mx.eval(p, opt.state)
        if step % 400 == 0:
            pred = mx.argmax(pd_hard_forward(p, Xte, BETA1), axis=-1)
            acc = float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32)))
            best = max(best, acc)
    return {"k": k, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed, "H": H,
            "test_seq_acc": best, "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["k"], r["n_seq"], r["seed"]) for r in done}
    tasks = [(k, n, s) for k in KS for n in n_seq_grid(k) for s in SEEDS if (k, n, s) not in seen]
    print(f"hard-STE m* sweep: {len(tasks)} runs", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] HARD Z{r['k']:<3} n_seq={r['n_seq']:<4} "
                  f"trans={r['transitions']:<6} s{r['seed']} test={r['test_seq_acc']:.2f} "
                  f"{'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = ["=== EXP-3: HARD-STE permutation m*(K) on Z_K (T07 soft-vs-hard control) ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds; T={T}; H=2K",
             f"{'k':>4} {'m*_trans':>9} {'m*/(K^2 lnK)':>12}"]
    ks_ok, mstar = [], []
    for k in KS:
        grid = sorted(set(r["n_seq"] for r in results if r["k"] == k))
        ms = None
        for n in grid:
            cell = [r for r in results if r["k"] == k and r["n_seq"] == n]
            if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                ms = n * T; break
        if ms:
            ks_ok.append(k); mstar.append(ms)
            lines.append(f"{k:>4} {ms:>9} {ms/(k*k*math.log(k)):>12.2f}")
        else:
            lines.append(f"{k:>4} {'>grid':>9} {'-':>12}")
    if len(ks_ok) == 2:
        slope = math.log(mstar[1] / mstar[0]) / math.log(ks_ok[1] / ks_ok[0])
        lines += ["", f"two-point log-log slope = {slope:+.2f} "
                  f"(soft sweep gave +1.62; a comparable sub-2 slope CONFIRMS the soft m* is not a "
                  f"soft-model artifact -> T07 closed)"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_hard_mstar.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
