"""MECHANISM of the group/random dichotomy: GD discovers the state HOMOMORPHISM on groups.

The paper shows GD generalizes from sub-coverage data on GROUP automata and fails on RANDOM ones, but
states the MECHANISM only as a conjecture ("the parameter-complexity separation is the natural
account"). This experiment measures the mechanism directly.

Claim under test: a gradient-trained recurrent model generalizes on a group exactly when its hidden
state COLLAPSES TO A FUNCTION OF THE GROUP ELEMENT -- i.e. it discovers the homomorphism phi: prefix
-> running-product-state, so two different prefixes that evaluate to the SAME group element map to
(nearly) the same hidden state. On a RANDOM automaton there is no algebra to compress, so the only way
to fit the training set is to MEMORIZE specific prefixes; fresh prefixes then do not collapse by state.

Metric -- homomorphism alignment A = SS_between(state) / SS_total of hidden states, grouped by the TRUE
running-product state y_t. A in [0,1]; A->1 means the representation is organized purely by the group
element (a clean homomorphism), A->0 means state identity is not linearly separable in the rep. We
measure A on BOTH the training prefixes (A_train) and FRESH held-out prefixes (A_fresh). The
memorization signature is the gap A_train - A_fresh: large for random (memorized, no OOD homomorphism),
small for groups (genuine homomorphism that transfers to fresh prefixes). cos_gap (within-state minus
between-state mean cosine on fresh prefixes) is a scale-free corroborating measure.

REGIME: trained at the DICHOTOMY budget (40x coverage) where the existing m* sweeps show GROUP
generalizes and RANDOM fails. (At this budget RANDOM cannot even fit the train set with H=2K, so its
A_train is also low; the separate "memorizes-train-but-fails-test" signature is a LOW-coverage
phenomenon probed in exp_mechanism_lo.py, not here.)

PRE-REGISTERED PREDICTIONS (logged BEFORE running the full sweep; a 3-cell smoke test informed the
regime calibration only):
  P1. Across all (arch, family, K) cells, A_fresh is strongly positively correlated with fresh test
      seq-acc (the representation organizing by group element is WHY generalization happens).
  P2. GROUP cells (Z cyclic, D dihedral non-abelian): high test-acc AND high A_fresh.
  P3. RANDOM cells: low test-acc AND low A_fresh (no homomorphism to discover).
  P4. The mechanism holds for the NON-ABELIAN group D too (not a cyclic-only artifact).
If instead random cells ALSO show high A_fresh, or group cells show low A_fresh, the homomorphism
account is FALSIFIED -> report honestly, do not fit a story to the data.

Architectures {pd, gru, lstm, rnn}; families {Z (cyclic), D (dihedral, non-abelian), RAND}; K matched
across families (Z12/D6, Z16/D8, Z24/D12; RAND12/16/24). Trained at fixed SUB-coverage so the dichotomy
is active. Idempotent per (arch,family,k,seed). Writes results/exp_mechanism.{json,txt}.
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_mechanism.json")

# family -> list of (group_name, K) with K matched across families
def cells_for_K(K):
    return {"Z": f"Z{K}", "D": f"D{K // 2}", "RAND": f"RAND{K}"}

KS = [16, 20, 24]
ARCHS = ["pd", "gru", "lstm", "rnn"]
SEEDS = [0, 1, 2]
T = 16
TRAIN_STEPS = 8000
LR = 0.003
GRAD_CLIP = 1.0
EVAL_FRESH = 512
PROBE = 256                  # fresh / train sequences used for the alignment probe
TRACE_STEPS = [1000, 2000, 4000, 6000, 8000]


def n_seq_for(K):
    # 40x coverage: the high-coverage regime where the EXISTING m* sweeps show GROUP generalizes (GRU
    # Z16 m*=1774 seq, Z20/24 fully) while RANDOM still FAILS (RAND best_test=0.00 even at 40x cov).
    # This is exactly the data budget that activates the dichotomy for the mechanism contrast.
    return round(40.0 * K * K * math.log(K) / T)


def alignment(Hnp, ynp):
    """A = SS_between / SS_total of rows of H grouped by label y. Hnp [N,d], ynp [N]."""
    import numpy as np
    mu = Hnp.mean(0)
    sst = float(((Hnp - mu) ** 2).sum())
    if sst <= 0:
        return 0.0
    ssb = 0.0
    for s in np.unique(ynp):
        Hs = Hnp[ynp == s]
        if len(Hs):
            ssb += len(Hs) * float(((Hs.mean(0) - mu) ** 2).sum())
    return ssb / sst


def cos_gap(Hnp, ynp):
    """mean within-state minus mean between-state cosine, using class-mean directions. Fresh data."""
    import numpy as np
    Hn = Hnp / (np.linalg.norm(Hnp, axis=1, keepdims=True) + 1e-8)
    states = np.unique(ynp)
    means = {s: Hn[ynp == s].mean(0) for s in states}
    M = np.stack([means[s] for s in states])                  # [S,d]
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    s_index = {s: i for i, s in enumerate(states)}
    sims = Hn @ Mn.T                                           # [N,S] cos(h_i, classmean_s)
    within = np.array([sims[i, s_index[ynp[i]]] for i in range(len(ynp))])
    # between = mean over s != y_i
    tot = sims.sum(1)
    btw = (tot - within) / (len(states) - 1)
    return float(within.mean() - btw.mean())


def worker(task):
    arch, fam, K, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    import models
    INIT = {"pd": models.pd_init, "gru": models.gru_init, "lstm": models.lstm_init, "rnn": models.rnn_init}
    FWD = {"pd": models.pd_forward, "gru": models.gru_forward, "lstm": models.lstm_forward, "rnn": models.rnn_forward}
    STA = {"pd": models.pd_states, "gru": models.gru_states, "lstm": models.lstm_states, "rnn": models.rnn_states}
    from models import loss_fn
    init, forward, states = INIT[arch], FWD[arch], STA[arch]

    group = cells_for_K(K)[fam]
    H = 2 * K
    n_seq = n_seq_for(K)
    data_rng = np.random.default_rng(1000 + seed)
    Xtr_np, Ytr_np = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr_np), mx.array(Ytr_np)
    te_rng = np.random.default_rng(7_000_000 + seed)
    Xte_np, Yte_np = make_batch(group, EVAL_FRESH, T, te_rng, stratified=False)
    Xte, Yte = mx.array(Xte_np), mx.array(Yte_np)
    # probe sets (subset of train; a fresh draw for fresh)
    Xtr_p, Ytr_p = Xtr[:PROBE], Ytr_np[:PROBE]
    pr_rng = np.random.default_rng(9_000_000 + seed)
    Xfr_np, Yfr_np = make_batch(group, PROBE, T, pr_rng, stratified=False)
    Xfr = mx.array(Xfr_np)

    def probe_A(Xp, Yp_np):
        hs = np.array(states(p, Xp))                # [N,T,H]
        Hflat = hs.reshape(-1, hs.shape[-1])
        yflat = Yp_np.reshape(-1)
        return alignment(Hflat, yflat), Hflat, yflat

    p = init(K, H, K, mx.random.key(seed)) if arch != "pd" else init(K, H, K, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(forward, pp, xx, yy))
    bs = min(64, n_seq)
    train_rng = np.random.default_rng(seed)
    best, best_tok = 0.0, 0.0
    trace = []
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
                best = s; best_tok = float(mx.mean((pred == Yte).astype(mx.float32)))
        if step in TRACE_STEPS:
            a_fr, _, _ = probe_A(Xfr, Yfr_np)
            trace.append({"step": step, "A_fresh": a_fr})

    tr = mx.argmax(forward(p, Xtr), axis=-1)
    train_acc = float(mx.mean(mx.all(tr == Ytr, axis=1).astype(mx.float32)))
    A_train, _, _ = probe_A(Xtr_p, Ytr_p)
    A_fresh, Hfr, yfr = probe_A(Xfr, Yfr_np)
    cg = cos_gap(Hfr, yfr)
    return {"arch": arch, "family": fam, "k": K, "group": group, "seed": seed, "H": H, "n_seq": n_seq,
            "transitions": n_seq * T, "train_seq_acc": train_acc, "test_seq_acc": best,
            "test_tok_acc": best_tok, "A_train": A_train, "A_fresh": A_fresh,
            "A_gap": A_train - A_fresh, "cos_gap_fresh": cg, "trace": trace}


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["arch"], d["family"], d["k"], d["seed"]) for d in done}
    tasks = [(a, f, k, s) for a in ARCHS for f in ["Z", "D", "RAND"] for k in KS for s in SEEDS
             if (a, f, k, s) not in seen]
    workers = 3
    print(f"mechanism probe: {len(tasks)} runs, {workers} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] {r['arch']:<4} {r['family']:<4} K={r['k']:<3} s{r['seed']} "
                  f"test={r['test_seq_acc']:.2f} A_fr={r['A_fresh']:.3f} A_tr={r['A_train']:.3f} "
                  f"gap={r['A_gap']:+.3f} cosgap={r['cos_gap_fresh']:+.3f}", flush=True)

    # summary: per (arch,family) medians; global correlation A_fresh vs test acc
    lines = ["=== MECHANISM: homomorphism alignment vs generalization ===",
             f"n={len(SEEDS)} seeds; T={T}; sub-coverage n_seq=4*K^1.6/T; A=SS_between(state)/SS_total"]
    allA, allAcc = [], []
    for arch in ARCHS:
        lines.append(f"\n[{arch}]")
        for fam in ["Z", "D", "RAND"]:
            for k in KS:
                cell = [r for r in results if r["arch"] == arch and r["family"] == fam and r["k"] == k]
                if not cell:
                    continue
                ta = median([r["test_seq_acc"] for r in cell])
                af = median([r["A_fresh"] for r in cell]); at = median([r["A_train"] for r in cell])
                cg = median([r["cos_gap_fresh"] for r in cell])
                allA.append(af); allAcc.append(ta)
                lines.append(f"   {fam:<4} K={k:<3} test={ta:.2f}  A_fresh={af:.3f} A_train={at:.3f} "
                             f"gap={at-af:+.3f} cos_gap={cg:+.3f}")
    if len(allA) >= 3:
        A = np.array(allA); Acc = np.array(allAcc)
        if A.std() > 0 and Acc.std() > 0:
            rho = float(np.corrcoef(A, Acc)[0, 1])
            lines.append(f"\n=== P1: corr(A_fresh, test_seq_acc) over {len(A)} cells = {rho:+.3f} ===")
    # group vs random aggregate
    def agg(fams):
        c = [r for r in results if r["family"] in fams]
        if not c:
            return None
        return (median([r["test_seq_acc"] for r in c]), median([r["A_fresh"] for r in c]),
                median([r["A_gap"] for r in c]))
    g = agg(["Z", "D"]); rd = agg(["RAND"])
    if g and rd:
        lines.append(f"GROUP(Z,D): test={g[0]:.2f} A_fresh={g[1]:.3f} A_gap={g[2]:+.3f}   "
                     f"RANDOM: test={rd[0]:.2f} A_fresh={rd[1]:.3f} A_gap={rd[2]:+.3f}")
        lines.append("P2/P3 hold if GROUP A_fresh high & gap small while RANDOM A_fresh low & gap large.")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_mechanism.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
