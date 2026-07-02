"""Read-only analyzer for the strengthening campaign (EXP-10 universality + EXP-11 mechanism).

Ingests results/exp_universality_{pd,gru,lstm,rnn}.json (where present) PLUS the existing
exp_gru_baseline.json / exp_extended_mstar.json as the PD/GRU reference, and results/exp_mechanism.json.
Robust to PARTIAL data: prints whatever has completed so far, so it doubles as a live monitor.

Produces, paper-ready:
  (A) Universality m*(K) table + log-log slope per architecture (PD, GRU, LSTM, RNN) on GROUP vs RANDOM,
      with the honest verdict (generalizes-sub-ERM / memorizes-but-fails / not-yet).
  (B) Mechanism: correlation of homomorphism-alignment A_fresh with fresh test seq-acc across ALL cells
      (P1), per-(arch,family) mean A_fresh and test (P2/P3), non-abelian D check (P4), and the
      train-time emergence trace of A.

No numbers are computed that are not directly in the result JSONs. Nothing is written; read-only.
Usage: python3 analyze_strengthen.py
"""
from __future__ import annotations
import os, json, math
from collections import defaultdict
from statistics import median, mean

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
THR = 0.95
KS_REF = [8, 12, 16, 20, 24, 32]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def slope_fit(ks, ms):
    if len(ks) < 3:
        return float("nan"), float("nan")
    import numpy as np
    lx, ly = np.log(ks), np.log(ms)
    s, a = np.polyfit(lx, ly, 1)
    pred = a + s * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(s), float(r2)


def mstar_table(records, family, ks=KS_REF):
    """m* = min transitions with median(seed) fresh-test seq-acc >= THR. Also best test per K."""
    out = {}
    for k in ks:
        grid = sorted(set(r["n_seq"] for r in records if r["family"] == family and r["k"] == k))
        mstar = None
        for n in grid:
            cell = [r for r in records if r["family"] == family and r["k"] == k and r["n_seq"] == n]
            if cell and median([r["test_seq_acc"] for r in cell]) >= THR:
                mstar = n * 16
                break
        cellall = [r for r in records if r["family"] == family and r["k"] == k]
        best_test = max([r["test_seq_acc"] for r in cellall], default=float("nan"))
        best_train = max([r["train_seq_acc"] for r in cellall], default=float("nan"))
        out[k] = (mstar, best_test, best_train)
    return out


def universality():
    print("=" * 78)
    print("(A) UNIVERSALITY  —  does the group/random dichotomy hold across architectures?")
    print("=" * 78)
    sources = {
        "PD-SSM": ("exp_extended_mstar.json", None),   # selective reference
        "GRU":    ("exp_gru_baseline.json", None),     # gated non-selective reference
        "LSTM":   ("exp_universality_lstm.json", None),
        "RNN":    ("exp_universality_rnn.json", None),
    }
    for label, (fname, _) in sources.items():
        recs = load(fname)
        if not recs:
            print(f"\n[{label}]  (no results file yet: {fname})")
            continue
        # normalize: exp_extended_mstar (PD-SSM Z-only) lacks 'family'/'train_seq_acc'
        for r in recs:
            r.setdefault("family", "Z")
            r.setdefault("train_seq_acc", float("nan"))
        # exp_extended_mstar has only Z (group). gru/lstm/rnn have Z + RAND.
        fams = sorted(set(r["family"] for r in recs))
        print(f"\n[{label}]  ({len(recs)} runs; families={fams})")
        for fam in [f for f in ("Z", "RAND") if f in fams]:
            tab = mstar_table(recs, fam)
            ks_ok = [k for k, (m, _, _) in tab.items() if m is not None]
            ms = [tab[k][0] for k in ks_ok]
            s, r2 = slope_fit(ks_ok, ms)
            famlabel = "GROUP(Z)" if fam == "Z" else "RANDOM"
            cells = "  ".join(
                f"K{k}:{'>grid' if tab[k][0] is None else tab[k][0]}"
                f"(t{tab[k][1]:.2f}/tr{tab[k][2]:.2f})" for k in KS_REF if k in tab)
            print(f"   {famlabel:9s} {cells}")
            if len(ks_ok) >= 3:
                print(f"             -> m* log-log slope = {s:+.2f} (r^2={r2:.3f}) over K={ks_ok}")
            else:
                gen = [k for k in KS_REF if tab.get(k, (None,))[0] is not None]
                # memorizes-but-fails diagnosis
                memo = [k for k in KS_REF if k in tab and tab[k][2] >= 0.95 and tab[k][1] < 0.5]
                if not gen and memo:
                    print(f"             -> NO m* at any K; MEMORIZES (train>=.95) but FAILS test at K={memo}")
                else:
                    print(f"             -> m* at K={gen} only (insufficient for slope)")
    print("\nReference exponents: tabular-ERM (structure-blind) slope ~2.13; K logK ~1.0; K^2 logK ~2.0.")
    print("Verdict logic: GROUP slope < ~1.9 with RANDOM failing at all K>=12  =>  arch shows the dichotomy.")


def mechanism():
    recs = load("exp_mechanism.json")
    print("\n" + "=" * 78)
    print("(B) MECHANISM  —  does homomorphism alignment A explain WHO generalizes?")
    print("=" * 78)
    if not recs:
        print("  (no results/exp_mechanism.json yet)")
        return
    print(f"  {len(recs)} cells completed.")
    # P1: correlation A_fresh vs test across ALL cells
    xs = [r["test_seq_acc"] for r in recs if "A_fresh" in r]
    ys = [r["A_fresh"] for r in recs if "A_fresh" in r]
    if len(xs) >= 3:
        import numpy as np
        c = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\n  P1  corr(A_fresh, test_seq_acc) over {len(xs)} cells = {c:+.3f}  "
              f"(predict strong positive)")
    # P2/P3/P4: per (arch, family) mean A_fresh & test
    print("\n  P2/P3/P4  mean over K,seed  [test_acc | A_fresh | A_train | A_gap | cos_gap]:")
    cellmap = defaultdict(list)
    for r in recs:
        cellmap[(r["arch"], r["family"])].append(r)
    print(f"   {'arch':5s}{'fam':5s}{'test':>7s}{'A_fr':>7s}{'A_tr':>7s}{'gap':>7s}{'cosg':>7s}{'n':>4s}")
    for arch in ["pd", "gru", "lstm", "rnn"]:
        for fam in ["Z", "D", "RAND"]:
            c = cellmap.get((arch, fam))
            if not c:
                continue
            def m(key):
                vals = [x[key] for x in c if key in x]
                return mean(vals) if vals else float("nan")
            print(f"   {arch:5s}{fam:5s}{m('test_seq_acc'):>7.2f}{m('A_fresh'):>7.3f}"
                  f"{m('A_train'):>7.3f}{m('A_gap'):>7.3f}{m('cos_gap_fresh'):>7.3f}{len(c):>4d}")
    # P4 explicit: non-abelian D present and group-like?
    dvals = [r for r in recs if r["family"] == "D"]
    if dvals:
        dtest = mean([r["test_seq_acc"] for r in dvals])
        dA = mean([r["A_fresh"] for r in dvals])
        print(f"\n  P4  NON-ABELIAN D: mean test={dtest:.2f}, mean A_fresh={dA:.3f}  "
              f"(predict group-like: high both)")
    # emergence trace
    traced = [r for r in recs if r.get("trace")]
    if traced:
        print("\n  A_fresh emergence over training steps (first few group vs random cells):")
        for r in traced[:6]:
            tr = [pt["A_fresh"] for pt in r["trace"]]
            seq = "  ".join(f"{v:.2f}" for v in tr)
            print(f"   {r['arch']:4s} {r['family']}{r['k']:<3} s{r['seed']}  [{seq}]  final_test={r['test_seq_acc']:.2f}")


if __name__ == "__main__":
    universality()
    mechanism()
    print("\n(analysis is read-only and partial-data-safe; re-run any time to monitor progress)")
