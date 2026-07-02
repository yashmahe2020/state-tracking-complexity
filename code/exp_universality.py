"""UNIVERSALITY of the group/random dichotomy across non-selective recurrent architectures.

The published paper's dichotomy D (GD beats the ERM rate on GROUP automata, fails on RANDOM ones)
currently has ONE non-selective control: the GRU (EXP-6). A single control cannot establish that the
effect is a property of gradient-trained recurrence in general rather than a GRU-specific quirk. This
experiment adds two more canonical non-selective recurrent controls -- a vanilla (Elman) RNN and an
LSTM -- run under the IDENTICAL protocol to the GRU m* sweep, so the dichotomy can be claimed
"across the recurrent architectures we test {GRU, RNN, LSTM}" rather than for the GRU alone.

Protocol (identical to exp_gru_baseline / m_star_sweep): for each architecture, family in
{Z (cyclic group), RAND (random automaton)} and each K, build a fixed training set of n_seq length-T
sequences, train (Adam lr 0.003 + global-norm grad-clip 1.0, 8000 steps, T=16, H=2K), evaluate
fresh-test sequence accuracy; m* = min transitions with median fresh-test seq-acc >= 0.95 across seeds.

PRE-REGISTERED INTERPRETATION (logged BEFORE running; mirrors ruling #5):
  * Per architecture, if group slope < 1.9 (sub-ERM) AND random FAILS (no m* in grid at any K>=12)
      -> that architecture exhibits the dichotomy. If ALL of {GRU,RNN,LSTM} do, D is UNIVERSAL across
         gradient-trained recurrence; selectivity contributes the diagonal separation S only.
  * if some architecture's RANDOM also generalizes -> the dichotomy is NOT universal; report honestly
     per-architecture and DEMOTE the universality claim (do not overstate).
  * if an architecture fails to optimize even the GROUP (train_seq_acc<0.95 at high coverage)
     -> that architecture is an invalid (non-optimizing) control at this budget; report and exclude,
        do not read its random-failure as evidence.

Self-contained + idempotent per (arch,family,k,n_seq,seed). Writes results/exp_universality_<arch>.{json,txt}.
Usage: python3 exp_universality.py <arch>   where arch in {rnn, lstm}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

FAMILIES = ["Z", "RAND"]
KS = [8, 12, 16, 20, 24, 32]
SEEDS = [0, 1, 2]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.003
GRAD_CLIP = 1.0
EVAL_FRESH = 512
MAX_WORKERS = 4

ARCHS = {"rnn": ("rnn_init", "rnn_forward"), "lstm": ("lstm_init", "lstm_forward")}


def n_seq_grid(k):
    import numpy as np
    cov = k * k * math.log(k) / T
    lo = max(3.0, 0.3 * k * math.log(k) / T)
    hi = 40.0 * cov
    g = np.unique(np.round(np.geomspace(lo, hi, 12)).astype(int))
    return [int(x) for x in g if x >= 1]


def worker(task):
    arch, family, k, n_seq, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    import models
    init = getattr(models, ARCHS[arch][0]); forward = getattr(models, ARCHS[arch][1])
    from models import loss_fn
    group = "%s%d" % (family, k)
    H = 2 * k
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(forward, pp, xx, yy))
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
            pred = mx.argmax(forward(p, Xte), axis=-1)
            s = float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32)))
            if s > best:
                best = s
                best_tok = float(mx.mean((pred == Yte).astype(mx.float32)))
    tr = mx.argmax(forward(p, Xtr), axis=-1)
    best_train = float(mx.mean(mx.all(tr == Ytr, axis=1).astype(mx.float32)))
    return {"arch": arch, "family": family, "k": k, "n_seq": n_seq, "transitions": n_seq * T,
            "seed": seed, "H": H, "train_seq_acc": best_train, "test_seq_acc": best,
            "test_tok_acc": best_tok, "generalizes": best >= THR}


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "rnn"
    assert arch in ARCHS, f"arch must be one of {list(ARCHS)}"
    os.makedirs(RES, exist_ok=True)
    OUT = os.path.join(RES, f"exp_universality_{arch}.json")
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["family"], d["k"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(arch, f, k, n, s) for f in FAMILIES for k in KS for n in n_seq_grid(k) for s in SEEDS
             if (f, k, n, s) not in seen]
    print(f"[{arch}] universality sweep: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] {arch} {r['family']}{r['k']:<3} n_seq={r['n_seq']:<5} "
                  f"trans={r['transitions']:<7} s{r['seed']} train={r['train_seq_acc']:.2f} "
                  f"test={r['test_seq_acc']:.2f} {'GEN' if r['generalizes'] else '.'}", flush=True)

    def mstar_and_slope(family):
        ks_ok, ms = [], []
        for k in KS:
            grid = sorted(set(r["n_seq"] for r in results if r["family"] == family and r["k"] == k))
            for n in grid:
                cell = [r for r in results if r["family"] == family and r["k"] == k and r["n_seq"] == n]
                if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                    ks_ok.append(k); ms.append(n * T); break
        slope = r2 = float("nan")
        if len(ks_ok) >= 3:
            lx, ly = np.log(ks_ok), np.log(ms)
            slope, a = np.polyfit(lx, ly, 1)
            pred = a + slope * lx
            r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        return ks_ok, ms, float(slope), float(r2)

    lines = [f"=== UNIVERSALITY: {arch.upper()} m*(K), group (Z) vs random (RAND) ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds; T={T}; H=2K",
             "(PD-SSM ref: Z slope +1.62; tabular-ERM slope 2.13; GRU ref: Z sub-ERM, RAND fails)"]
    verdict = {}
    for fam in FAMILIES:
        ks_ok, ms, slope, r2 = mstar_and_slope(fam)
        verdict[fam] = (ks_ok, ms, slope, r2)
        lines.append(f"\n[{fam}]  m* per k:")
        for k in KS:
            m = dict(zip(ks_ok, ms)).get(k)
            cell = [r for r in results if r["family"] == fam and r["k"] == k]
            bestmax = max([r["test_seq_acc"] for r in cell], default=float("nan"))
            lines.append(f"   k={k:>3}  m*={'>grid' if m is None else m:>8}  best_test@maxcov={bestmax:.2f}")
        lines.append(f"   -> log-log slope = {slope:+.2f} (r^2={r2:.3f}) over k={ks_ok}"
                     if len(ks_ok) >= 3 else f"   -> insufficient m* points (gen at k={ks_ok})")
    zks, zms, zslope, zr2 = verdict["Z"]; rks, rms, rslope, rr2 = verdict["RAND"]
    rand_gen = len([k for k in rks if k >= 12]) > 0
    group_beats = (len(zks) >= 3) and (zslope < 1.9)
    lines.append("\n=== PRE-REGISTERED INTERPRETATION ===")
    if not rand_gen and group_beats:
        lines.append(f"{arch.upper()} EXHIBITS the dichotomy (group slope {zslope:+.2f}<1.9, random fails) "
                     "-> consistent with a UNIVERSAL recurrent-bias effect.")
    elif rand_gen:
        lines.append(f"{arch.upper()} GENERALIZES on RANDOM (m* in grid for some k>=12) -> dichotomy NOT "
                     "universal for this arch; report honestly, do not overstate.")
    elif not group_beats:
        lines.append(f"{arch.upper()} does not beat ERM on groups (slope>=1.9) -> selectivity/recurrence "
                     "distinction inconclusive for this arch at this budget.")
    else:
        lines.append("Mixed/insufficient signal -> inspect raw curve.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, f"exp_universality_{arch}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
