"""EXP-4 (Phase-4 review, advisor ruling #4, addresses triage T15 / pre-registered confound C4): the
paper claims the ERM sample complexity is counted in TRANSITIONS and is independent of sequence length
T (permutation transitions have operator norm 1, so no e^{O(T)} blow-up). This was argued but never
swept. Here we fix K=16 (Z_16) and measure the statistical m* (in transitions) at T in {8,16,32}; the
prediction is m* ~ constant in T (a flat coefficient of variation), confirming m* counts transitions
not sequences or steps. Soft PD-SSM, same protocol as m_star_sweep. Writes results/exp_tsweep.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_tsweep.json")

K = 16
TS = [8, 16, 32]
SEEDS = [0, 1, 2]
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.01
EVAL_FRESH = 512
MAX_WORKERS = 3


def n_seq_grid(k, T):
    # grid in SEQUENCES chosen so the transition count n_seq*T spans the same range across T
    import numpy as np
    cov_trans = k * k * math.log(k)          # bare coverage scale in transitions
    lo_trans = max(3.0 * T, 0.3 * k * math.log(k))
    hi_trans = 40.0 * cov_trans
    g_trans = np.unique(np.round(np.geomspace(lo_trans, hi_trans, 12)).astype(int))
    g_seq = np.unique(np.maximum(1, np.round(g_trans / T)).astype(int))
    return [int(x) for x in g_seq]


def worker(task):
    T, n_seq, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_init, pd_forward, loss_fn
    group = "Z%d" % K
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    H = 2 * K
    p = pd_init(K, H, K, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(pd_forward, pp, xx, yy))
    bs = min(64, n_seq)
    train_rng = np.random.default_rng(seed)
    best = 0.0
    for step in range(1, TRAIN_STEPS + 1):
        idx = train_rng.integers(0, n_seq, size=bs)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        opt.update(p, g); mx.eval(p, opt.state)
        if step % 250 == 0:
            pred = mx.argmax(pd_forward(p, Xte), axis=-1)
            acc = float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32)))
            best = max(best, acc)
    return {"T": T, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed,
            "test_seq_acc": best, "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["T"], r["n_seq"], r["seed"]) for r in done}
    tasks = [(T, n, s) for T in TS for n in n_seq_grid(K, T) for s in SEEDS if (T, n, s) not in seen]
    print(f"T-sweep m* (K={K}): {len(tasks)} runs", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] T={r['T']:<3} n_seq={r['n_seq']:<4} "
                  f"trans={r['transitions']:<6} s{r['seed']} test={r['test_seq_acc']:.2f} "
                  f"{'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = [f"=== EXP-4: T-independence of m* (transitions) at K={K} (T15 / confound C4) ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds",
             f"{'T':>4} {'m*_trans':>9}"]
    ts_ok, mstar = [], []
    for T in TS:
        grid = sorted(set(r["n_seq"] for r in results if r["T"] == T))
        ms = None
        for n in grid:
            cell = [r for r in results if r["T"] == T and r["n_seq"] == n]
            if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                ms = n * T; break
        if ms:
            ts_ok.append(T); mstar.append(ms)
            lines.append(f"{T:>4} {ms:>9}")
        else:
            lines.append(f"{T:>4} {'>grid':>9}")
    if len(mstar) >= 2:
        arr = np.array(mstar, float)
        cv = float(arr.std() / arr.mean())
        lines += ["", f"m*(transitions) across T: {mstar}, CV={cv:.3f}",
                  f"VERDICT: {'T-INDEPENDENT (CV<0.30): m* counts transitions, not sequences/steps -> C4 closed.' if cv < 0.30 else 'T-DEPENDENT (CV>=0.30): m* varies with T -> inspect / disclose.'}"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_tsweep.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
