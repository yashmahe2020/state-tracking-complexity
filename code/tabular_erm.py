"""PHASE 2C-THEOREM: optimization-free confirmation of the ERM sample complexity Theta(K^2 log K).

The theorem is about ANY consistent learner (ERM), not a particular optimizer. The cleanest realization
is a TABULAR learner: observe m labeled transitions (cells (state, symbol) -> next state); it knows
delta on observed cells and is wrong on unobserved ones. Tested on FRESH length-T sequences, a sequence
is correct iff ALL T of its transition cells were observed in training. m* = min m with seq-acc >= THR.
This is PURE COVERAGE -> coupon-collector over the K^2 reachable cells -> Theta(K^2 log K), and it is
identical for structured (Z_k) and unstructured (random) targets because a tabular learner exploits NO
structure. This (a) confirms the theorem's order and tightness empirically with ZERO optimization
confound, and (b) gives the ERM baseline curve against which the GD-PD-SSM's sub-bound rate on Z_k
(K^1.6, structure discovery) is the contribution. No MLX / no training -- pure combinatorics.

DISCRIMINATOR: m*/(K^2 log K) flat (CV < 0.15) AND log-log slope in [1.7, 2.3] CONFIRMS Theta(K^2 log K).
"""
from __future__ import annotations
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "tabular_erm.json")

KS = [8, 12, 16, 20, 24, 32, 48, 64]
FAMILIES = ["Z", "RAND"]      # cyclic group vs random automaton -- tabular m* must MATCH (structure-blind)
SEEDS = list(range(8))         # average m* over data-sampling seeds (cheap)
T = 16
THR = 0.95
EVAL_FRESH = 2000


def reachable_cells(table, k):
    """Cells (state, symbol) reachable from start state 0 under length-T random walks. With the
    full-symbol alphabet and one-step mixing, every (state, symbol) pair is reachable for t>=1, so
    the cell universe is all K*K pairs."""
    return k * k


def measure_mstar(family, k, seed):
    import sys; sys.path.insert(0, HERE)
    from groups import get_group, running_product
    table, kk = get_group(f"{family}{k}")
    assert kk == k
    rng = np.random.default_rng(seed)
    # FRESH test sequences (fixed per seed): each is a list of (state,symbol) cells visited
    Xte = rng.integers(0, k, size=(EVAL_FRESH, T), dtype=np.int64)
    Yte = running_product(Xte, table)
    # state BEFORE each step: state_{t-1}; start = 0
    prev = np.concatenate([np.zeros((EVAL_FRESH, 1), np.int64), Yte[:, :-1]], axis=1)  # [N,T] state before step t
    test_cells = prev * k + Xte                      # cell id per (seq,t)
    # sweep training size m (in transitions); observe m random cells along random training walks
    grid = sorted(set(int(round(x)) for x in np.geomspace(0.1 * k * k, 25 * k * k * math.log(k), 40)))
    mstar = None
    for m in grid:
        n_seq = max(1, m // T)
        Xtr = rng.integers(0, k, size=(n_seq, T), dtype=np.int64)
        Ytr = running_product(Xtr, table)
        ptr = np.concatenate([np.zeros((n_seq, 1), np.int64), Ytr[:, :-1]], axis=1)
        seen = np.zeros(k * k, dtype=bool)
        seen[(ptr * k + Xtr).ravel()] = True
        # a fresh seq is correct iff all its cells were observed
        ok = seen[test_cells].all(axis=1)            # [N]
        if ok.mean() >= THR:
            mstar = n_seq * T
            break
    return mstar


def main():
    os.makedirs(RES, exist_ok=True)
    from statistics import median
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["family"], r["k"], r["seed"]) for r in results}
    for fam in FAMILIES:
        for k in KS:
            for s in SEEDS:
                if (fam, k, s) in seen:
                    continue
                m = measure_mstar(fam, k, s)
                results.append({"family": fam, "k": k, "seed": s, "mstar_trans": m})
                json.dump(results, open(OUT, "w"), indent=2)
                print(f"  {fam}{k:<3} s{s} m*={m}", flush=True)

    def cv(xs):
        xs = np.array(xs, float)
        return float(xs.std() / xs.mean()) if len(xs) >= 2 and xs.mean() > 0 else float("nan")

    lines = ["=== TABULAR ERM sample complexity m*(K) — pure coverage, optimization-free ===",
             f"m* = min transitions s.t. fresh-test (all-cells-observed) seq-acc >= {THR}; T={T}; "
             f"median over {len(SEEDS)} seeds",
             f"{'fam':>5} {'k':>4} {'m*_trans':>9} {'m*/(K^2 lnK)':>12} {'m*/(KlnK)':>10}"]
    for fam in FAMILIES:
        ks, ms = [], []
        for k in KS:
            cell = [r["mstar_trans"] for r in results if r["family"] == fam and r["k"] == k and r["mstar_trans"]]
            if not cell:
                continue
            m = median(cell); ks.append(k); ms.append(m)
            lines.append(f"{fam:>5} {k:>4} {m:>9.0f} {m/(k*k*math.log(k)):>12.2f} {m/(k*math.log(k)):>10.1f}")
        if len(ks) >= 3:
            c2 = [m/(k*k*math.log(k)) for k, m in zip(ks, ms)]
            c1 = [m/(k*math.log(k)) for k, m in zip(ks, ms)]
            lx, ly = np.log(ks), np.log(ms); b, a = np.polyfit(lx, ly, 1)
            r2 = 1 - np.sum((ly - (a + b*lx))**2) / np.sum((ly - ly.mean())**2)
            tight = cv(c2) < 0.15 and 1.7 <= b <= 2.3
            lines += [f"   -> [{fam}] m*/(K^2 lnK) CV={cv(c2):.3f}  m*/(K lnK) CV={cv(c1):.3f}  "
                      f"slope={b:+.2f} (r2={r2:.3f})  => {'CONFIRMS Theta(K^2 logK)' if tight else 'NOT tight K^2'}",
                      ""]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "tabular_erm.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
