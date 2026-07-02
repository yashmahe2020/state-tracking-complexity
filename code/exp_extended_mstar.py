"""EXP-7 (Phase-4, advisor ruling #5): extended-K PD-SSM statistical m* sweep on cyclic groups.

The Phase-2B m* sweep used 6 K values (8..32) at 3 seeds and gave slope +1.62 (r^2 0.91) with a
NON-MONOTONE point at K=32 and the pre-registered constant-flatness discriminator INCONCLUSIVE
(CV 0.27 / 0.39). Round-3 JMLR reviewers flagged the two-decimal slope as an overclaim of precision.
This run extends the grid to K in {8..48} and raises the seed count to 5, to either FIRM the slope or
honestly qualify it.

PRE-REGISTERED DECISION RULE (ledger ruling #5): report a numeric slope ONLY if r^2 >= 0.97 AND the
fitted-constant CV < 0.20 across the extended range; otherwise retreat to the qualitative claim
"m* is systematically below the Theta(K^2 logK) baseline (slope 2.13)" and DROP the two-decimal slope.

Identical protocol to m_star_sweep (SOFT PD-SSM, fixed training set, fresh-test seq-acc>=0.95, T=16,
Adam lr 0.01, 8000 steps, H=2K). Self-contained + idempotent per (k,n_seq,seed). Writes
results/exp_extended_mstar.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_extended_mstar.json")

KS = [8, 12, 16, 20, 24, 32, 40, 48]
SEEDS = [0, 1, 2, 3, 4]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.01
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
    from models import pd_init, pd_forward, loss_fn
    group = "Z%d" % k
    H = 2 * k
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = pd_init(k, H, k, mx.random.key(seed))
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
            best = max(best, float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32))))
    return {"k": k, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed, "H": H,
            "test_seq_acc": best, "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["k"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(k, n, s) for k in KS for n in n_seq_grid(k) for s in SEEDS if (k, n, s) not in seen]
    print(f"extended m* sweep: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] Z{r['k']:<3} n_seq={r['n_seq']:<5} trans={r['transitions']:<7} "
                  f"s{r['seed']} test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = ["=== EXP-7: extended-K PD-SSM statistical m*(K) on Z_k (5 seeds, K to 48) ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds; T={T}; H=2K",
             f"{'k':>4} {'m*_trans':>9} {'m*/(KlnK)':>10} {'m*/(K^2 lnK)':>12}"]
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
            lines.append(f"{k:>4} {ms:>9} {ms/(k*math.log(k)):>10.1f} {ms/(k*k*math.log(k)):>12.2f}")
        else:
            lines.append(f"{k:>4} {'>grid':>9} {'-':>10} {'-':>12}")

    def cv(xs):
        xs = np.array(xs, float); return float(xs.std() / xs.mean()) if len(xs) >= 2 and xs.mean() > 0 else float("nan")
    if len(ks_ok) >= 3:
        lx, ly = np.log(ks_ok), np.log(mstar)
        b, a = np.polyfit(lx, ly, 1)
        pred = a + b * lx
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        c1 = [m / (k * math.log(k)) for k, m in zip(ks_ok, mstar)]
        c2 = [m / (k * k * math.log(k)) for k, m in zip(ks_ok, mstar)]
        cv1, cv2 = cv(c1), cv(c2)
        lines += ["", f"K logK constant CV={cv1:.3f}; K^2 logK constant CV={cv2:.3f}",
                  f"log-log slope = {b:+.2f} (r^2={r2:.3f}) over K={ks_ok}"]
        # pre-registered decision rule (ruling #5)
        clean = (r2 >= 0.97) and (min(cv1, cv2) < 0.20)
        if clean:
            lines.append(f"DECISION (ruling #5): CLEAN — report slope {b:+.2f} (r^2 {r2:.3f}, CV {min(cv1,cv2):.3f}).")
        else:
            lines.append(f"DECISION (ruling #5): NOT CLEAN (r^2 {r2:.3f} < 0.97 or CV {min(cv1,cv2):.3f} >= 0.20) "
                         f"-> RETREAT to qualitative 'm* systematically below the Theta(K^2 logK) baseline "
                         f"(slope 2.13)'; DROP the two-decimal slope from abstract/claims.")
    else:
        lines.append("DECISION: INSUFFICIENT m* points.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_extended_mstar.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
