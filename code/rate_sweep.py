"""PHASE 2 (validate) -- the C2 rate experiment: measure the sample-complexity EXPONENT of PD-SSM.

Committed claim: PD-SSM learns a K-state permutation automaton to EXACT state tracking with
Theta(K^2 log K) labeled transitions. Pre-registered decision rule: fit log(transitions-to-grok) vs
log(K) on cyclic Z_k (clean K control); CONFIRM iff log-log slope in [1.7, 2.3] with r^2 >= 0.95;
slope ~1 falsifies (-> rate is K log K, revise); unstable/r^2<0.95 -> consult advisor.

CONFOUND CONTROLS baked in (ledger section 2):
  [C1] CAPACITY ablation: run BOTH H=2k (capacity ~k^3) and H=const (capacity ~k). If the slope agrees
       across H-modes, the exponent is the STATISTICAL rate, not a capacity-scaling artifact.
  [C2] grok criterion = SEQUENCE accuracy (ALL T positions correct), matching the exact-tracking
       theorem -- NOT mean per-position accuracy.
  [C3] cell coverage: tokens are uniform over the FULL group (k symbols) -> the visited-state marginal
       is uniform for every t>=1 in any finite group (one-step mixing); coupon-collector over K^2 cells
       holds without start randomization. (Verified separately by verify_uniform_marginal.)
  decoupled seeds: data rng = init seed + 100000.
  fine grok detection: evaluate every EVAL_EVERY steps.
  disclosure logging: parameter count, OOD position-accuracy at 4T, softmax entropy of P_logits at grok.

K is swept on Z_k only (the controlled-K, diag-learnable family). The qualitative diag-vs-PD separation
is the SEPARATE non-solvable S5 result (c2_viability). T is held fixed (T-independence tested by the
companion t_independence.py). Writes results/rate_sweep.{json,txt}. Idempotent per (k,Hmode,seed).
"""
from __future__ import annotations
import os, sys, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "rate_sweep.json")

KS = [4, 6, 8, 12, 16, 24, 32]   # cyclic Z_k; batch*T/k^2 >= 1 up to k=32 (coverage adequate)
H_CONST = 64                      # >= max K, so P_logits ~ k*H^2 = O(k) capacity (decouples from k)
SEEDS = [0, 1, 2, 3, 4]
LR = 0.01
BATCH = 64
T = 16
BUDGET = 20000
EVAL_EVERY = 25
GROK = 0.95          # SEQUENCE accuracy threshold (exact tracking)
EVAL_B = 256
MAX_WORKERS = 4


def worker(task):
    k, hmode, seed = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_hard_init, pd_hard_forward, loss_fn
    fwd = pd_hard_forward

    group = "Z%d" % k
    H = (2 * k) if hmode == "scaled" else H_CONST
    p = pd_hard_init(k, H, k, mx.random.key(seed))
    n_params = int(sum(np.prod(np.array(v).shape) for v in p.values()))
    opt = optim.Adam(learning_rate=LR)
    rng = np.random.default_rng(seed)                 # data rng ...
    eval_rng = np.random.default_rng(seed + 100_000)  # ... DECOUPLED from init seed
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(fwd, pp, xx, yy))

    def seq_acc(xx, yy):
        pred = mx.argmax(fwd(p, xx), axis=-1)          # [B,T]
        return float(mx.mean(mx.all(pred == yy, axis=1).astype(mx.float32)))

    def pos_acc(xx, yy):
        pred = mx.argmax(fwd(p, xx), axis=-1)
        return float(mx.mean((pred == yy).astype(mx.float32)))

    xe, ye = make_batch(group, EVAL_B, T, eval_rng)
    xe, ye = mx.array(xe), mx.array(ye)
    xo, yo = make_batch(group, EVAL_B, 4 * T, eval_rng)   # OOD length 4T (length generalization)
    xo, yo = mx.array(xo), mx.array(yo)

    # grok = EXACT tracking that LENGTH-GENERALIZES: OOD (4T) sequence accuracy >= GROK. This certifies
    # the true automaton was learned (a fixed-length fit fails OOD), matching the exact-tracking theorem.
    grok_step = None
    train_at_grok = ood_pos_at_grok = float("nan")
    for step in range(1, BUDGET + 1):
        x, y = make_batch(group, BATCH, T, rng)
        _, g = lg(p, mx.array(x), mx.array(y))
        opt.update(p, g)
        mx.eval(p, opt.state)
        if step % EVAL_EVERY == 0 or step == 1:
            if seq_acc(xo, yo) >= GROK:
                grok_step = step
                train_at_grok = seq_acc(xe, ye)
                ood_pos_at_grok = pos_acc(xo, yo)
                break
    final_ood_seq = seq_acc(xo, yo)
    transitions = (grok_step * BATCH * T) if grok_step else None
    return {"k": k, "hmode": hmode, "H": H, "seed": seed, "n_params": n_params,
            "grok_step": grok_step, "transitions_to_grok": transitions,
            "final_ood_seq_acc": final_ood_seq, "train_seq_at_grok": train_at_grok,
            "ood_pos_acc_at_grok": ood_pos_at_grok, "budget": BUDGET, "T": T, "batch": BATCH, "lr": LR}


def key(d):
    return (d["k"], d["hmode"], d["seed"])


def loglog_fit(ks, ys):
    import numpy as np
    ks = np.array(ks, float); ys = np.array(ys, float)
    m = ys > 0
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(ks[m]), np.log(ys[m])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ssr = np.sum((ly - pred) ** 2); sst = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    return float(b), float(r2)


def main():
    os.makedirs(RES, exist_ok=True)
    import numpy as np
    from statistics import median
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {key(d) for d in done}
    tasks = [(k, hm, s) for k in KS for hm in ("scaled", "const") for s in SEEDS if (k, hm, s) not in seen]
    print(f"rate sweep: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
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
            print(f"  [{i}/{len(tasks)}] Z{r['k']:<3} H={r['hmode']:<6}(H={r['H']}) s{r['seed']} "
                  f"grok@{r['grok_step']} trans={r['transitions_to_grok']} "
                  f"oodseq={r['final_ood_seq_acc']:.2f} params={r['n_params']}", flush=True)

    # ---- per-(hmode,k) medians + log-log fit per hmode ----
    lines = ["=== PD-SSM sample-complexity rate sweep (Z_k, sequence-accuracy grok) ===",
             f"grok = SEQUENCE acc >= {GROK}; transitions = grok_step * batch({BATCH}) * T({T}); n={len(SEEDS)} seeds",
             f"{'hmode':>6} {'k':>4} {'n_grok':>7} {'med_trans':>11} {'med_step':>9} {'med_params':>11} {'med_oodpos':>10}"]
    fit = {}
    for hm in ("scaled", "const"):
        ks_fit, tr_fit = [], []
        for k in KS:
            rs = [r for r in results if r["k"] == k and r["hmode"] == hm]
            gr = [r for r in rs if r["transitions_to_grok"]]
            if gr:
                mt = median([r["transitions_to_grok"] for r in gr])
                ms = median([r["grok_step"] for r in gr])
                ks_fit.append(k); tr_fit.append(mt)
            else:
                mt = ms = None
            mp = median([r["n_params"] for r in rs]) if rs else None
            mo = median([r["ood_pos_acc_at_grok"] for r in gr]) if gr else float("nan")
            lines.append(f"{hm:>6} {k:>4} {f'{len(gr)}/{len(rs)}':>7} {str(mt):>11} {str(ms):>9} "
                         f"{str(mp):>11} {mo:>10.3f}")
        slope, r2 = loglog_fit(ks_fit, tr_fit)
        fit[hm] = (slope, r2, len(ks_fit))
        lines.append(f"  -> {hm}: log-log slope = {slope:+.2f}, r^2 = {r2:.4f} (n_k={len(ks_fit)})")
        lines.append("")

    ss, rs2, _ = fit.get("scaled", (float('nan'),)*3)
    cs, rc2, _ = fit.get("const", (float('nan'),)*3)
    # K^2 log K predicts slope ~2 (slightly above due to the log factor); pre-registered band [1.7,2.3]
    in_band = lambda s: 1.7 <= s <= 2.3
    cap_free = (not math.isnan(ss) and not math.isnan(cs) and abs(ss - cs) <= 0.4)
    if in_band(ss) and in_band(cs) and rs2 >= 0.95 and rc2 >= 0.95 and cap_free:
        verdict = (f"CONFIRMED: slope scaled={ss:+.2f} (r2={rs2:.3f}), const={cs:+.2f} (r2={rc2:.3f}); both "
                   f"in [1.7,2.3] and agree within 0.4 -> the exponent is the STATISTICAL rate K^2 log K, "
                   f"NOT a capacity artifact (const-H has O(k) capacity yet same slope). Proceed to write.")
    elif (not math.isnan(ss)) and 0.7 <= ss <= 1.3 and 0.7 <= cs <= 1.3:
        verdict = (f"FALSIFIED toward K log K: slope scaled={ss:+.2f}, const={cs:+.2f} ~ 1. The committed "
                   f"K^2 log K rate is wrong; REVISE claim to Theta(K log K) and re-derive the upper bound "
                   f"(permutation closure lets GD generalize from O(K log K) cells). Consult advisor.")
    else:
        verdict = (f"INCONCLUSIVE / consult advisor: slope scaled={ss:+.2f} (r2={rs2:.3f}), const={cs:+.2f} "
                   f"(r2={rc2:.3f}); cap_free={cap_free}. Not cleanly in a band -> inspect per-k, check lr "
                   f"plateau / coverage at large k before ruling.")
    lines += ["PRE-REGISTERED VERDICT: " + verdict]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "rate_sweep.txt"), "w").write(txt + "\n")
    print(f"\n-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
