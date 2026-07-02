"""DE-CONFOUND the selective-vs-GRU data-efficiency edge: PD-SSM m*(K) at the GRU's learning rate.

The frozen-v18 paper reports the selective(PD-SSM)-vs-GRU data-efficiency gap (m* ~9 vs ~19 at K=32)
as "optimizer-inclusive", because the published PD-SSM m* sweep ran at lr0.01 while the GRU baseline
ran at lr0.003 (a GRU diverges at lr0.01). That makes the gap confound inductive bias with optimizer.
This experiment removes the confound: it re-measures PD-SSM m*(K) on GROUP (Z) and RANDOM automata at
lr0.003 — IDENTICAL to exp_gru_baseline — so PD and GRU are compared at a MATCHED optimizer.

Protocol = exp_gru_baseline.py verbatim except the model is PD-SSM (pd_init/pd_forward) and we keep
lr0.003 + global-norm grad-clip 1.0. m* = min transitions with median(seed) fresh-test seq-acc >= 0.95.

PRE-REGISTERED INTERPRETATION (logged BEFORE running):
  * If at matched lr0.003 the PD-SSM GROUP m* still sits BELOW the GRU GROUP m* (e.g. K=32 PD < GRU's
      ~68640 transitions / 19.3 normalized), the selective data-efficiency edge SURVIVES matched
      optimization -> it is a genuine architectural effect, and the v19 abstract/caption can DROP the
      optimizer-confound hedge for the matched comparison.
  * If PD and GRU m* CONVERGE at matched lr (within seed noise), the edge was optimizer-driven -> report
      the HONEST null: the dichotomy (group sub-ERM, random fails) holds for both, but the selective vs
      gated RANKING is not robust to the optimizer. Either outcome strengthens the paper's rigor.
  * If PD-SSM fails to optimize the group at lr0.003 (train<0.95 at high coverage) -> lr0.003 is a
      non-optimizing budget for PD; report and compare only where both optimize.

Self-contained + idempotent per (family,k,n_seq,seed). Writes results/exp_pd_lr003.{json,txt}.
Usage: python3 exp_pd_lr003.py
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_pd_lr003.json")

FAMILIES = ["Z", "RAND"]
KS = [8, 12, 16, 20, 24, 32]
SEEDS = [0, 1, 2]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.003           # MATCHED to exp_gru_baseline (the whole point)
GRAD_CLIP = 1.0
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
    family, k, n_seq, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_init, pd_forward, loss_fn
    group = "%s%d" % (family, k)
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
    best, best_tok = 0.0, 0.0
    for step in range(1, TRAIN_STEPS + 1):
        idx = train_rng.integers(0, n_seq, size=bs)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        gnorm = mx.sqrt(sum(mx.sum(v * v) for v in g.values()))
        scale = mx.minimum(1.0, GRAD_CLIP / (gnorm + 1e-6))
        g = {kk: vv * scale for kk, vv in g.items()}
        opt.update(p, g); mx.eval(p, opt.state)
        if step % 250 == 0:
            pred = mx.argmax(pd_forward(p, Xte), axis=-1)
            s = float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32)))
            if s > best:
                best = s
                best_tok = float(mx.mean((pred == Yte).astype(mx.float32)))
    tr = mx.argmax(pd_forward(p, Xtr), axis=-1)
    best_train = float(mx.mean(mx.all(tr == Ytr, axis=1).astype(mx.float32)))
    return {"arch": "pd_lr003", "family": family, "k": k, "n_seq": n_seq, "transitions": n_seq * T,
            "seed": seed, "H": H, "train_seq_acc": best_train, "test_seq_acc": best,
            "test_tok_acc": best_tok, "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["family"], d["k"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(f, k, n, s) for f in FAMILIES for k in KS for n in n_seq_grid(k) for s in SEEDS
             if (f, k, n, s) not in seen]
    print(f"[pd_lr003] de-confound sweep: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] pd_lr003 {r['family']}{r['k']:<3} n_seq={r['n_seq']:<5} "
                  f"trans={r['transitions']:<7} s{r['seed']} train={r['train_seq_acc']:.2f} "
                  f"test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    def mstar(family):
        ks_ok, ms = [], []
        for k in KS:
            for n in sorted(set(r["n_seq"] for r in results if r["family"] == family and r["k"] == k)):
                cell = [r for r in results if r["family"] == family and r["k"] == k and r["n_seq"] == n]
                if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                    ks_ok.append(k); ms.append(n * T); break
        return ks_ok, ms

    lines = ["=== DE-CONFOUND: PD-SSM m*(K) at lr0.003 (MATCHED to GRU) ===",
             "Compare to GRU lr0.003 group m*(trans): >grid/>grid/28384/47936/37600/68640 (K=8..32),",
             "and PD-SSM lr0.01 group m*(trans): 3472/8528/8896/25584/19312/33200 (slope +1.62)."]
    for fam in FAMILIES:
        ks_ok, ms = mstar(fam)
        slope = r2 = float("nan")
        if len(ks_ok) >= 3:
            lx, ly = np.log(ks_ok), np.log(ms)
            slope, a = np.polyfit(lx, ly, 1)
            pred = a + slope * lx
            r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        lines.append(f"\n[{fam}] m* per k:")
        for k in KS:
            m = dict(zip(ks_ok, ms)).get(k)
            cell = [r for r in results if r["family"] == fam and r["k"] == k]
            bestmax = max([r["test_seq_acc"] for r in cell], default=float("nan"))
            norm = (m / (k * k * math.log(k))) if m else float("nan")
            lines.append(f"   k={k:>3}  m*={'>grid' if m is None else m:>8}  best_test@maxcov={bestmax:.2f}"
                         f"  m*/(K^2 lnK)={norm:.2f}")
        lines.append(f"   -> log-log slope = {slope:+.2f} (r^2={r2:.3f}) over k={ks_ok}"
                     if len(ks_ok) >= 3 else f"   -> insufficient m* points (gen at k={ks_ok})")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_pd_lr003.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
