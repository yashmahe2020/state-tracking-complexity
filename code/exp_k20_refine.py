"""Refinement probe (Phase-4 review, JMLR audit item): pin m* at K=20 (cyclic) on a FINE n_seq grid
with MORE seeds, to test whether the non-monotonic spike (m*=25584 at K=20, above K=16's 8896 and
K=24's 19312 in exp_extended_mstar) is a grid-resolution artifact rather than a real effect.

The extended sweep used a 12-point geometric grid whose adjacent points near K=20 were ~13616 and
~25536 transitions, so any true m* in (13616, 25536] was rounded UP to 25536. This probe fills that
gap at n_seq in {850..1600} (transitions ~13600..25600) and raises seeds 5 -> 8.

PROTOCOL identical to exp_extended_mstar (SOFT PD-SSM, fixed training set, fresh-test seq-acc>=0.95,
T=16, Adam lr 0.01, 8000 steps, H=2K). data_rng = 1000+seed and model key = seed, EXACTLY as the
original worker, so seeds 0..4 reproduce the original runs and 5..7 are genuinely new. Idempotent per
(k,n_seq,seed). NO claim depends on the outcome (the paper already retreats to the qualitative
sub-baseline statement); this only sharpens the reported K=20 value and the monotonicity disclosure.
Writes results/exp_k20_refine.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_k20_refine.json")

K = 20
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_SEQ_GRID = [850, 1000, 1150, 1300, 1450, 1600]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.01
EVAL_FRESH = 512
MAX_WORKERS = 3


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
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["k"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(K, n, s) for n in N_SEQ_GRID for s in SEEDS if (K, n, s) not in seen]
    print(f"K=20 refine: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] Z{r['k']} n_seq={r['n_seq']:<5} trans={r['transitions']:<6} "
                  f"s{r['seed']} test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = ["=== K=20 refinement: fine n_seq grid, 8 seeds (was 5), SOFT PD-SSM Z_20 ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds; T={T}; H=2K",
             f"original extended-sweep value (5 seeds, coarse grid): m*=25584",
             f"{'n_seq':>6} {'trans':>7} {'med_acc':>8} {'n_seeds':>8}  per-seed"]
    mstar = None
    for n in N_SEQ_GRID:
        cell = [r for r in results if r["k"] == K and r["n_seq"] == n]
        if not cell:
            continue
        accs = sorted(r["test_seq_acc"] for r in cell)
        med = median(accs)
        lines.append(f"{n:>6} {n*T:>7} {med:>8.3f} {len(cell):>8}  {['%.2f'%a for a in accs]}")
        if mstar is None and med >= THR:
            mstar = n * T
    lines.append("")
    if mstar:
        lines.append(f"REFINED m*(K=20, 8 seeds) = {mstar} transitions "
                     f"(was 25584; {'LOWER -> non-monotonicity was a grid artifact' if mstar < 25584 else 'CONFIRMED'}).")
    else:
        lines.append("REFINED m*(K=20): still > grid top (25600); spike not a grid artifact.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_k20_refine.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
