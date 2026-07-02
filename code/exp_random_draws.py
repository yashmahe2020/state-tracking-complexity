"""EXP-2 (Phase-4 review, advisor ruling #4, addresses triage T12): the necessity result (Thm 4,
random K-automaton needs H >= K) is PROBABILISTIC over the draw of the random automaton, but the
Phase-2C min-H table used a SINGLE draw per K. This re-runs the min-H probe for several INDEPENDENT
random-automaton draws per K, to confirm the "GD-unlearnable at H <= 2K" verdict is typical of the
random class, not an artifact of one lucky/unlucky draw.

For each K in {12,16,20} and each automaton draw seed in {0,1,2}, sweep H up to 2K and report the
smallest H whose fresh-test sequence accuracy >= 0.95 (or ">2K" = GD-unlearnable in budget). The
random automaton for draw d is RAND<K>s<d> (groups.random_perm_table seed). Coverage is made
non-limiting (N_SEQ=3000 >> coverage m*). Writes results/exp_random_draws.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_random_draws.json")

KS = [12, 16, 20]
DRAWS = [0, 1, 2]          # independent random-automaton draws (RAND<k>s<draw>)
T = 16
THR = 0.95
STEPS = 12000
LR = 0.01
N_SEQ = 3000               # >> coverage, so H is the only bottleneck
EVAL_FRESH = 512
MAX_WORKERS = 3


def h_grid(k):
    base = [2, 3, 4, 6, 8]
    scaled = [int(round(c * k)) for c in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    return sorted(set(h for h in base + scaled if 2 <= h <= 2 * k + 1))


def worker(task):
    k, draw, H = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_init, pd_forward, loss_fn
    group = "RAND%ds%d" % (k, draw)        # independent random automaton per draw
    dr = np.random.default_rng(1000 + draw)
    Xtr, Ytr = make_batch(group, N_SEQ, T, dr, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    te = np.random.default_rng(7_000_000 + draw)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, te, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = pd_init(k, H, k, mx.random.key(draw))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(pd_forward, pp, xx, yy))
    rng = np.random.default_rng(draw)
    best = 0.0
    for s in range(1, STEPS + 1):
        idx = rng.integers(0, N_SEQ, size=64)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        opt.update(p, g); mx.eval(p, opt.state)
        if s % 400 == 0:
            acc = float(mx.mean(mx.all(mx.argmax(pd_forward(p, Xte), axis=-1) == Yte, axis=1).astype(mx.float32)))
            best = max(best, acc)
    return {"k": k, "draw": draw, "H": H, "test_seq_acc": best, "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["k"], r["draw"], r["H"]) for r in done}
    tasks = [(k, d, H) for k in KS for d in DRAWS for H in h_grid(k) if (k, d, H) not in seen]
    print(f"random-draws min-H: {len(tasks)} runs", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] RAND{r['k']:<3} draw{r['draw']} H={r['H']:<3} "
                  f"test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    lines = ["=== EXP-2: min-H for INDEPENDENT random-automaton draws (T12 robustness) ===",
             f"min-H = smallest H with fresh-test seq-acc >= {THR}; H grid up to 2K; N_SEQ={N_SEQ}; steps={STEPS}",
             ">2K means GD-unlearnable within the H<=2K budget (the Thm-4 necessity prediction)",
             f"{'k':>4} {'draw':>5} {'min_H':>7} {'best_acc@maxH':>14}"]
    unlearnable = 0
    total = 0
    for k in KS:
        for d in DRAWS:
            total += 1
            grid = h_grid(k)
            found = None
            for H in grid:
                cell = [r for r in results if r["k"] == k and r["draw"] == d and r["H"] == H]
                if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                    found = H; break
            best_at_max = max([r["test_seq_acc"] for r in results
                               if r["k"] == k and r["draw"] == d], default=float("nan"))
            if found:
                lines.append(f"{k:>4} {d:>5} {found:>7} {best_at_max:>14.2f}")
            else:
                unlearnable += 1
                lines.append(f"{k:>4} {d:>5} {'>2K':>7} {best_at_max:>14.2f}")
    lines += ["", f"VERDICT: {unlearnable}/{total} (K,draw) cells GD-unlearnable at H<=2K. "
              f"{'ROBUST: random necessity holds across draws (not a single-draw artifact).' if unlearnable >= total - 1 else 'MIXED: some draws learnable -> inspect.'}"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_random_draws.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
