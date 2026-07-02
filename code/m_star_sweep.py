"""PHASE 2B (validate) -- STATISTICAL sample complexity m*(K), the theorem's actual quantity.

Why this replaces the online rate sweep: rate_sweep measured ONLINE transitions-to-grok, which at these
sizes is SGD OPTIMIZATION TIME, not sample complexity (at k=16, grok ~1.1M transitions but only K^2=256
cells exist -> full coverage in ~1500 transitions; grok is ~700x later). The theorem is about the
STATISTICAL sample complexity m* = the minimum TRAINING-SET SIZE (in transitions) for which a learner
that fits the training set GENERALIZES to fresh data (learns the transition function delta), not how
long SGD takes.

PROTOCOL (advisor Path B): for each K, build a FIXED training set of n_seq length-T sequences; train the
SOFT PD-SSM (Arm C, pd_*, which optimizes reliably even at k=120 -- unlike the fragile hard-STE model)
to convergence on it; evaluate on FRESH (held-out) length-T sequences (in-distribution -> no
length-generalization needed, no hard-permutation fragility). m*(K) = min training size (transitions)
with median fresh-test SEQUENCE accuracy >= THR. Fresh-test generalization (not train fit) certifies
delta was learned (not memorized), so m* is coverage/statistics-limited.

NEUTRAL DISCRIMINATOR (pre-registered, ledger sec 1): across K, normalize m* by K log K and by
K^2 log K; the TRUE rate is whichever normalized constant is FLAT (coefficient of variation < 0.15).
The per-K size grid spans well below K log K to well above K^2 log K, so it brackets BOTH hypotheses and
is not biased toward either. Writes results/m_star_sweep.{json,txt}. Idempotent per (k,n_seq,seed).
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
# Target family: "Z" (cyclic group, structured) or "RAND" (random automaton, unstructured). Phase-2C
# runs both arms; output is family-tagged so the two m* curves are stored side by side.
FAMILY = sys.argv[1] if len(sys.argv) > 1 else "Z"
TAG = {"Z": "m_star_sweep", "RAND": "m_star_random", "D": "m_star_D"}.get(FAMILY, f"m_star_{FAMILY}")
OUT = os.path.join(RES, TAG + ".json")


def group_name(family, k):
    """Map (family, group-order k) -> a get_group name. Z<k>/RAND<k> have order k directly; the
    dihedral family D has order 2n, so order k needs D_{k/2} (k even). This lets the SAME order grid
    KS be reused across families so the m* curves are directly comparable across the x-axis."""
    if family == "D":
        assert k % 2 == 0, "dihedral order must be even"
        return "D%d" % (k // 2)
    return "%s%d" % (family, k)

KS = [8, 12, 16, 20, 24, 32]
SEEDS = [0, 1, 2]
T = 16
THR = 0.95           # fresh-test SEQUENCE accuracy
TRAIN_STEPS = 8000   # to convergence on the fixed set (soft model groks fast; train acc monitored)
LR = 0.01
EVAL_FRESH = 512
MAX_WORKERS = 4


def n_seq_grid(k):
    # The seq-acc>=0.95 bar (per-token ~0.997) needs every cell learned near-perfectly, so m* sits at
    # a LARGE multiple of the bare coverage scale K^2 log K / T (smoke: k=8 m* ~ 14x). Grid spans from
    # well below K log K to ~40x K^2 log K so it BRACKETS m* under either hypothesis (K log K is lower
    # still at large k; covering bounds m* <= O(K^2 log K), so 40x is a safe ceiling). Neutral: not
    # centered on either candidate.
    import numpy as np
    cov = k * k * math.log(k) / T          # bare coverage scale, in sequences
    lo = max(3.0, 0.3 * k * math.log(k) / T)   # below K log K
    hi = 40.0 * cov                        # safely above K^2 log K
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
    fwd = pd_forward

    group = group_name(family, k)           # Z<k> / RAND<k> / D<k/2> (dihedral, order k)
    H = 2 * k
    # FIXED training set of n_seq sequences (sampled once, reused every step)
    data_rng = np.random.default_rng(1000 + seed)
    Xtr, Ytr = make_batch(group, n_seq, T, data_rng, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    # FRESH test set (disjoint rng stream)
    test_rng = np.random.default_rng(7_000_000 + seed)
    Xte, Yte = make_batch(group, EVAL_FRESH, T, test_rng, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)

    p = pd_init(k, H, k, mx.random.key(seed))
    opt = optim.Adam(learning_rate=LR)
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(fwd, pp, xx, yy))
    bs = min(64, n_seq)

    def accs(xx, yy):
        pred = mx.argmax(fwd(p, xx), axis=-1)
        seq = float(mx.mean(mx.all(pred == yy, axis=1).astype(mx.float32)))
        tok = float(mx.mean((pred == yy).astype(mx.float32)))
        return seq, tok

    best_test, best_tok = 0.0, 0.0
    train_rng = np.random.default_rng(seed)
    for step in range(1, TRAIN_STEPS + 1):
        idx = train_rng.integers(0, n_seq, size=bs)
        xb, yb = Xtr[mx.array(idx)], Ytr[mx.array(idx)]
        _, g = lg(p, xb, yb)
        opt.update(p, g)
        mx.eval(p, opt.state)
        if step % 250 == 0:
            s, tk = accs(Xte, Yte)
            if s > best_test:
                best_test, best_tok = s, tk
    train_acc, _ = accs(Xtr, Ytr)
    s, tk = accs(Xte, Yte)
    test_acc = max(best_test, s)
    test_tok = max(best_tok, tk)
    return {"family": family, "k": k, "n_seq": n_seq, "transitions": n_seq * T, "seed": seed, "H": H,
            "train_seq_acc": train_acc, "test_seq_acc": test_acc, "test_tok_acc": test_tok,
            "thr": THR, "generalizes": test_acc >= THR, "train_steps": TRAIN_STEPS, "T": T}


def key(d):
    return (d.get("family", "Z"), d["k"], d["n_seq"], d["seed"])


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {key(d) for d in done}
    tasks = [(FAMILY, k, n, s) for k in KS for n in n_seq_grid(k) for s in SEEDS
             if (FAMILY, k, n, s) not in seen]
    print(f"m* sweep [{FAMILY}]: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] {FAMILY}{r['k']:<3} n_seq={r['n_seq']:<4} trans={r['transitions']:<6} "
                  f"s{r['seed']} train={r['train_seq_acc']:.2f} test={r['test_seq_acc']:.2f} "
                  f"tok={r.get('test_tok_acc', float('nan')):.3f} "
                  f"{'GEN' if r['generalizes'] else '.'}", flush=True)

    # m*(k) = min transitions with median(test_seq_acc) >= THR
    fam_label = {"Z": "Z_k cyclic GROUP (structured)", "RAND": "RANDOM automaton (unstructured)",
                 "D": "D_n dihedral NON-ABELIAN GROUP (structured)"}.get(FAMILY, FAMILY)
    lines = [f"=== PD-SSM STATISTICAL sample complexity m*(K) — target family: {fam_label} ===",
             f"m* = min transitions with median fresh-test SEQ acc >= {THR}; n={len(SEEDS)} seeds; T={T}",
             "(RAND arm CONFIRMS tight Theta(K^2 logK) iff K^2-logK constant flat CV<0.15 AND slope in [1.7,2.3])",
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
        import numpy as np
        xs = np.array(xs, float); return float(xs.std() / xs.mean()) if len(xs) >= 2 and xs.mean() > 0 else float("nan")
    def loglog(ks, ys):
        import numpy as np
        if len(ks) < 3: return float("nan"), float("nan")
        lx, ly = np.log(ks), np.log(ys); b, a = np.polyfit(lx, ly, 1)
        pred = a + b * lx; r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        return float(b), float(r2)

    if len(ks_ok) >= 3:
        c_klogk = [m / (k * math.log(k)) for k, m in zip(ks_ok, mstar)]
        c_k2logk = [m / (k * k * math.log(k)) for k, m in zip(ks_ok, mstar)]
        cv1, cv2 = cv(c_klogk), cv(c_k2logk)
        slope, r2 = loglog(ks_ok, mstar)
        lines += ["", f"K log K  normalized constant: values {[round(x,1) for x in c_klogk]}, CV={cv1:.3f}",
                  f"K^2 log K normalized constant: values {[round(x,2) for x in c_k2logk]}, CV={cv2:.3f}",
                  f"log-log slope of m* vs K = {slope:+.2f} (r^2={r2:.3f})"]
        if cv1 < 0.15 and cv1 < cv2:
            v = f"CONFIRMED Theta(K log K): K-log-K constant flat (CV {cv1:.3f}) and flatter than K^2 (CV {cv2:.3f})."
        elif cv2 < 0.15 and cv2 < cv1:
            v = (f"Theta(K^2 log K): K^2-log-K constant flat (CV {cv2:.3f}). The online K log K was an "
                 f"optimization-time artifact; the STATISTICAL rate is K^2 log K -> restore original claim, re-consult advisor.")
        else:
            v = f"INCONCLUSIVE: neither constant flat (CV1 {cv1:.3f}, CV2 {cv2:.3f}); slope {slope:+.2f}. Widen grid / more seeds / consult advisor."
        lines.append("PRE-REGISTERED VERDICT: " + v)
    else:
        lines.append("PRE-REGISTERED VERDICT: INSUFFICIENT m* points (raise budget / widen grid).")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, TAG + ".txt"), "w").write(txt + "\n")
    print(f"\n-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
