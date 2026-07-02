"""PHASE 2C-MECHANISM: why GD-PD-SSM generalizes on structured but not unstructured targets.

Hypothesis (the mechanism behind the dichotomy): algebraically structured automata admit a LOW-
DIMENSIONAL state representation that GD's implicit bias finds (abelian -> commuting transitions ->
simultaneously diagonalizable -> the dynamics live in a ~O(1)-dim rotating frame), whereas a random
automaton has no compressible representation, so fitting the data forces a FULL ~K-dimensional
(memorizing) representation that does not generalize. Diagnostic: train the soft PD-SSM to train-fit on
matched (K, H, data, steps) for Z_k vs RAND_k, then measure the EFFECTIVE RANK (participation ratio
PR = (sum lambda)^2 / sum(lambda^2)) of the covariance of the visited hidden states. Prediction:
PR(Z_k) is LOW and ~flat in K; PR(RAND_k) is HIGH and grows ~linearly with K. A systematic
PR(Z) << PR(RAND) at matched K supports the structure-discovery mechanism (advisor's JMLR gate).
Writes results/rank_diag.{json,txt}.
"""
from __future__ import annotations
import os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "rank_diag.json")

KS = [12, 16, 20, 24]
FAMILIES = ["Z", "RAND"]
SEEDS = [0, 1, 2]
T = 16
H = 64
STEPS = 20000
LR = 0.005
N_SEQ = 3000          # >> coverage for both; Z generalizes, RAND memorizes
N_PROBE = 256         # sequences over which to collect visited states
MAX_WORKERS = 3


def worker(task):
    family, k, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_init, pd_forward, loss_fn
    dr = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(f"{family}{k}", N_SEQ, T, dr, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    te = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(f"{family}{k}", 512, T, te, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = pd_init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(pd_forward, pp, xx, yy))
    rng = np.random.default_rng(seed)
    for s in range(STEPS):
        idx = rng.integers(0, N_SEQ, size=64)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        opt.update(p, g); mx.eval(p, opt.state)

    def sacc(xx, yy):
        return float(mx.mean(mx.all(mx.argmax(pd_forward(p, xx), axis=-1) == yy, axis=1).astype(mx.float32)))
    tr_acc, te_acc = sacc(Xtr[:512], Ytr[:512]), sacc(Xte, Yte)

    # collect visited hidden states over probe sequences (replicate pd_forward's scan, keep states)
    P = mx.softmax(p["P_logits"], axis=-1); d = mx.tanh(p["d_raw"]); M = P * d[:, None, :]
    xb = Xtr[:N_PROBE]
    h = mx.broadcast_to(p["h0"], (N_PROBE, H)); states = []
    for t in range(T):
        h = mx.einsum("bij,bj->bi", M[xb[:, t]], h); states.append(h)
    Hs = np.array(mx.stack(states, axis=1)).reshape(-1, H)        # [N_PROBE*T, H]
    Hs = Hs - Hs.mean(0, keepdims=True)
    cov = (Hs.T @ Hs) / Hs.shape[0]
    ev = np.linalg.eigvalsh(cov); ev = np.clip(ev, 0, None)
    pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))        # participation ratio (effective rank)
    # also effective rank of the stacked transition operators {M(g)} (how many independent ops)
    Mflat = np.array(M).reshape(k, H * H); Mflat = Mflat - Mflat.mean(0, keepdims=True)
    sv = np.linalg.svd(Mflat, compute_uv=False); sv2 = sv ** 2
    op_pr = float((sv2.sum() ** 2) / (np.sum(sv2 ** 2) + 1e-12))
    return {"family": family, "k": k, "seed": seed, "train_acc": tr_acc, "test_acc": te_acc,
            "state_pr": pr, "op_pr": op_pr, "H": H}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["family"], r["k"], r["seed"]) for r in done}
    tasks = [(f, k, s) for f in FAMILIES for k in KS for s in SEEDS if (f, k, s) not in seen]
    print(f"rank_diag: {len(tasks)} runs", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] {r['family']}{r['k']:<3} s{r['seed']} "
                  f"train={r['train_acc']:.2f} test={r['test_acc']:.2f} "
                  f"state_PR={r['state_pr']:.1f} op_PR={r['op_pr']:.1f}", flush=True)

    lines = ["=== MECHANISM: effective dimensionality of the learned representation (Z_k vs random) ===",
             f"state_PR = participation ratio of visited-state covariance; op_PR = eff. rank of {{M(g)}}; "
             f"H={H}; median n={len(SEEDS)}",
             f"{'fam':>5} {'k':>4} {'test_acc':>8} {'state_PR':>9} {'op_PR':>7}"]
    tab = {}
    for f in FAMILIES:
        for k in KS:
            cell = [r for r in results if r["family"] == f and r["k"] == k]
            if not cell:
                continue
            spr = median([r["state_pr"] for r in cell]); opr = median([r["op_pr"] for r in cell])
            ta = median([r["test_acc"] for r in cell]); tab[(f, k)] = (spr, opr, ta)
            lines.append(f"{f:>5} {k:>4} {ta:>8.2f} {spr:>9.1f} {opr:>7.1f}")
    lines.append("")
    common = [k for k in KS if ("Z", k) in tab and ("RAND", k) in tab]
    if common:
        ratios = [tab[("RAND", k)][0] / max(tab[("Z", k)][0], 1e-9) for k in common]
        lines.append(f"state_PR ratio RAND/Z per K {common}: {[round(x,1) for x in ratios]}")
        zpr = [tab[("Z", k)][0] for k in common]; rpr = [tab[("RAND", k)][0] for k in common]
        import numpy as np
        z_slope = float(np.polyfit(np.log(common), np.log(zpr), 1)[0]) if len(common) >= 3 else float("nan")
        r_slope = float(np.polyfit(np.log(common), np.log(rpr), 1)[0]) if len(common) >= 3 else float("nan")
        lines.append(f"state_PR log-log slope vs K:  Z={z_slope:+.2f} (expect ~0, low-dim)  "
                     f"RAND={r_slope:+.2f} (expect ~1, grows with K)")
        if min(ratios) > 2.0 and r_slope > 0.5:
            lines.append("VERDICT: MECHANISM SUPPORTED — structured solutions are low-dimensional & flat in K; "
                         "random solutions are high-dimensional & grow with K. GD finds the compressible "
                         "representation only when one exists (structure).")
        else:
            lines.append("VERDICT: mechanism NOT clearly supported (PR gap small or random PR flat) -> "
                         "be conservative in the paper; the gap is unexplained.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "rank_diag.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
