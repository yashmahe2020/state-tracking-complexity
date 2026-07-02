"""EXP-6 (Phase-4, advisor ruling #5): NON-SELECTIVE GRU baseline for the group/random dichotomy.

Two independent round-3 JMLR reviewers rated this P0: the paper attributes the dichotomy (GD beats the
ERM rate on GROUP automata, fails on RANDOM automata) to the SELECTIVE inductive bias of the PD-SSM,
but ran no non-selective recurrent control. A standard GRU is the canonical control: its recurrent
transition is the SAME for every input (the token enters only additively via a learned embedding), so
it has no input-selected transition matrix. If the GRU shows the SAME dichotomy, the effect is a
property of gradient-trained recurrence, not selectivity specifically.

Protocol is identical to the PD-SSM m* sweep (m_star_sweep / exp_extended_mstar): for each family in
{Z (cyclic group), RAND (random automaton)} and each K, build a fixed training set of n_seq length-T
sequences, train the GRU (H=2K, Adam lr 0.003 + global-norm grad-clip 1.0, 8000 steps, T=16) on it, evaluate fresh-test sequence
accuracy; m* = min transitions with median fresh-test seq-acc >= 0.95 across seeds. H=2K matches the
PD-SSM sweep exactly so the comparison is apples-to-apples.

PRE-REGISTERED INTERPRETATION RULE (ledger ruling #5, logged BEFORE this run):
  * if GRU-group slope < 1.9 AND GRU-random FAILS to generalize (no m* within grid at any K>=12)
      -> the dichotomy is a RECURRENT-bias effect; reframe D accordingly (selectivity then contributes
         only the diagonal separation S). T1, T2 unaffected.
  * if GRU-random ALSO generalizes (random m* within grid) -> the dichotomy is NOT robust to the
      learner; DEMOTE D to a preliminary observation.
  * if GRU-group does NOT beat ERM (group slope >= ~2 / no sub-baseline) while PD-SSM does
      -> selectivity IS the discriminating bias; the original selective-bias framing STANDS and
         strengthens.

Self-contained + idempotent per (family,k,n_seq,seed). Writes results/exp_gru_baseline.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_gru_baseline.json")

FAMILIES = ["Z", "RAND"]
KS = [8, 12, 16, 20, 24, 32]
SEEDS = [0, 1, 2]
T = 16
THR = 0.95
TRAIN_STEPS = 8000
LR = 0.003          # was 0.01: a GRU at lr 0.01 with no grad clip DIVERGES at K>=16 (train_seq=0.00,
                    # loss oscillating); a smoke test confirmed lr 0.003 + clip 1.0 reaches train_seq=1.0
                    # by step 2000. A valid (optimizing) baseline is required for a fair selectivity test.
GRAD_CLIP = 1.0     # global-norm gradient clipping
EVAL_FRESH = 512
MAX_WORKERS = 5     # GRU models are tiny; raised from 3 to use idle headroom (91% CPU idle, 73GB free)


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
    from models import gru_init, gru_forward, loss_fn
    group = "%s%d" % (family, k)             # Z<k> / RAND<k> (random draw seed 0)
    H = 2 * k
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    p = gru_init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(gru_forward, pp, xx, yy))
    bs = min(64, n_seq)
    train_rng = np.random.default_rng(seed)
    best, best_tok, best_train = 0.0, 0.0, 0.0
    for step in range(1, TRAIN_STEPS + 1):
        idx = train_rng.integers(0, n_seq, size=bs)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        gnorm = mx.sqrt(sum(mx.sum(v * v) for v in g.values()))   # global-norm clip (GRU stability)
        scale = mx.minimum(1.0, GRAD_CLIP / (gnorm + 1e-6))
        g = {kk: vv * scale for kk, vv in g.items()}
        opt.update(p, g); mx.eval(p, opt.state)
        if step % 250 == 0:
            pred = mx.argmax(gru_forward(p, Xte), axis=-1)
            s = float(mx.mean(mx.all(pred == Yte, axis=1).astype(mx.float32)))
            if s > best:
                best = s
                best_tok = float(mx.mean((pred == Yte).astype(mx.float32)))
    tr = mx.argmax(gru_forward(p, Xtr), axis=-1)
    best_train = float(mx.mean(mx.all(tr == Ytr, axis=1).astype(mx.float32)))
    return {"family": family, "k": k, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed, "H": H,
            "train_seq_acc": best_train, "test_seq_acc": best, "test_tok_acc": best_tok,
            "generalizes": best >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["family"], d["k"], d["n_seq"], d["seed"]) for d in done}
    tasks = [(f, k, n, s) for f in FAMILIES for k in KS for n in n_seq_grid(k) for s in SEEDS
             if (f, k, n, s) not in seen]
    print(f"GRU baseline sweep: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] GRU {r['family']}{r['k']:<3} n_seq={r['n_seq']:<5} "
                  f"trans={r['transitions']:<7} s{r['seed']} train={r['train_seq_acc']:.2f} "
                  f"test={r['test_seq_acc']:.2f} tok={r['test_tok_acc']:.2f} "
                  f"{'GEN' if r['generalizes'] else '.'}", flush=True)

    def mstar_and_slope(family):
        ks_ok, ms = [], []
        for k in KS:
            grid = sorted(set(r["n_seq"] for r in results if r["family"] == family and r["k"] == k))
            found = None
            for n in grid:
                cell = [r for r in results if r["family"] == family and r["k"] == k and r["n_seq"] == n]
                if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                    found = n * T; break
            if found:
                ks_ok.append(k); ms.append(found)
        slope = r2 = float("nan")
        if len(ks_ok) >= 3:
            lx, ly = np.log(ks_ok), np.log(ms)
            slope, a = np.polyfit(lx, ly, 1)
            pred = a + slope * lx
            r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        return ks_ok, ms, float(slope), float(r2)

    lines = ["=== EXP-6: NON-SELECTIVE GRU baseline m*(K), group (Z) vs random (RAND) ===",
             f"m* = min transitions with median fresh-test seq-acc >= {THR}; n={len(SEEDS)} seeds; T={T}; H=2K",
             "(PD-SSM ref: Z slope +1.62; tabular-ERM slope 2.13; PD-SSM random GD-UNLEARNABLE)"]
    verdict = {}
    for fam in FAMILIES:
        ks_ok, ms, slope, r2 = mstar_and_slope(fam)
        verdict[fam] = (ks_ok, ms, slope, r2)
        lines.append(f"\n[{fam}]  m* per k:")
        for k in KS:
            m = dict(zip(ks_ok, ms)).get(k)
            # best test acc at max coverage (for the RAND fail/pass read)
            cell = [r for r in results if r["family"] == fam and r["k"] == k]
            bestmax = max([r["test_seq_acc"] for r in cell], default=float("nan"))
            lines.append(f"   k={k:>3}  m*={'>'+'grid' if m is None else m:>8}  best_test@maxcov={bestmax:.2f}")
        if len(ks_ok) >= 3:
            lines.append(f"   -> log-log slope = {slope:+.2f} (r^2={r2:.3f}) over k={ks_ok}")
        else:
            lines.append(f"   -> insufficient m* points (generalizes at k={ks_ok})")

    # auto interpretation per ruling #5
    zks, zms, zslope, zr2 = verdict["Z"]
    rks, rms, rslope, rr2 = verdict["RAND"]
    rand_generalizes = len([k for k in rks if k >= 12]) > 0
    group_beats_erm = (len(zks) >= 3) and (zslope < 1.9)
    lines.append("\n=== PRE-REGISTERED INTERPRETATION (ruling #5) ===")
    if not rand_generalizes and group_beats_erm:
        lines.append("GRU shows the SAME dichotomy (group sub-baseline slope<1.9, random fails) -> the "
                     "dichotomy is a RECURRENT-bias effect, NOT selective-specific. REFRAME D as "
                     "recurrent-bias; selectivity contributes the diagonal separation S. T1,T2 unaffected.")
    elif rand_generalizes:
        lines.append("GRU GENERALIZES on RANDOM automata (m* within grid for some k>=12) -> the dichotomy "
                     "is NOT robust to the learner. DEMOTE D to a preliminary observation; re-consult advisor.")
    elif not group_beats_erm:
        lines.append("GRU does NOT beat the ERM baseline on groups (slope >= ~1.9 / no sub-baseline) while "
                     "PD-SSM does -> SELECTIVITY is the discriminating bias. Original selective-bias framing "
                     "STANDS and strengthens.")
    else:
        lines.append("Mixed/insufficient signal -> inspect raw curve and consult advisor.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_gru_baseline.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
