"""C2 PIVOT VIABILITY GATE (advisor-pre-registered, pre-commitment).

Candidate C2 (GO from advisor): the sample complexity of learning a K-state FSA with a
discrete-selection SSM (PD-SSM) is Theta(K log K) -- O(K log K) covering-number upper bound, matching
Omega(K log K) Fano/packing lower bound (standard SLT, NOT the intractable diagonal-SSM RMT wall).
The empirical leg of the gate: the bound is non-vacuous only if the discrete selection is GENUINELY
NEEDED, i.e. on a non-solvable group a diagonal (commuting) SSM CANNOT solve the task while the
selective PD-SSM CAN. If a diagonal SSM also solves it, the "separation" is an optimization artifact
and the covering/packing story collapses -> KILL.

PRE-REGISTERED DECISION RULE (advisor, this fork):
  Run {S3 (non-abelian, k=6), S5 (NON-SOLVABLE, k=120)} x {diag (Arm A, commuting), pd (Arm C,
  selective)}, n=6 seeds. SEPARATION CONFIRMED on a group iff
     median(pd final_train_acc) - median(diag final_train_acc) >= 0.40  AND  median(pd) >= 0.85.
  The DEPTH-ROBUST (JMLR-load-bearing) case is S5: a non-solvable group cannot be tracked by any
  constant-depth diagonal SSM (Merrill TC0/NC1), so a diag failure there is representational, not an
  optimization artifact.
  GATE PASSES iff separation CONFIRMED on S5 (and, as a sanity check, on S3).
  GATE FAILS (KILL / re-hunt) iff diag also solves S5 (median diag >= 0.85), i.e. no expressiveness gap.

Reuses train.run_experiment verbatim (arms diag / pd already implemented and tested). Writes
results/c2_viability.{json,txt}. Idempotent (skips finished (group,arm,seed) cells).
"""
from __future__ import annotations
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "c2_viability.json")
BAR = 0.85
MAX_WORKERS = 4
SEEDS = [0, 1, 2, 3, 4, 5]

# (group, arm, T_train, batch, lr, budget). FAIRNESS: identical optimizer (lr, batch) per arm; the
# diagonal arm is given MORE steps than PD-SSM so any diag failure is representational, not
# under-training. PD-SSM solves S5 (k=120) by ~1000 steps at lr=0.01/batch=64 (smoke-verified).
CONFIGS = [
    ("S3", "diag", 48, 64, 0.01, 15000),
    ("S3", "pd",   48, 64, 0.01,  8000),
    ("S5", "diag", 16, 64, 0.01, 15000),
    ("S5", "pd",   16, 64, 0.01,  8000),
]


def worker(task):
    group, arm, T, batch, lr, budget, seed = task
    sys.path.insert(0, HERE)
    from train import run_experiment
    fast = (arm == "diag")   # diag has a machine-verified vectorized forward
    r = run_experiment(group, arm, seed, budget, lr=lr, batch=batch,
                       T_train=T, T_ood=T, log_every=max(200, budget // 50), fast=fast)
    h = r["history"]
    steps, acc = h["step"], h["train_acc"]
    grok = next((steps[i] for i, a in enumerate(acc) if a >= BAR), None)
    return {"group": group, "arm": arm, "k": r["k"], "T": T, "batch": batch, "lr": lr,
            "budget": budget, "seed": seed,
            "final_acc": r["final_train_acc"], "max_acc": max(acc), "grok_step": grok,
            "final_loss": r["final_loss"], "wall_sec": r["wall_sec"]}


def key(d):
    return (d["group"], d["arm"], d["seed"])


def main():
    os.makedirs(RES, exist_ok=True)
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {key(d) for d in done}
    tasks = [(g, a, T, bs, lr, bud, s) for (g, a, T, bs, lr, bud) in CONFIGS
             for s in SEEDS if (g, a, s) not in seen]
    print(f"C2 viability gate: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r)
            json.dump(results, open(OUT, "w"))
            print(f"  [{i}/{len(tasks)}] {r['group']}/{r['arm']} s{r['seed']} "
                  f"grok@{r['grok_step']} final={r['final_acc']:.3f} max={r['max_acc']:.3f} "
                  f"({r['wall_sec']:.0f}s)", flush=True)

    # ---- verdict ----
    cell = {}
    for r in results:
        cell.setdefault((r["group"], r["arm"]), []).append(r)
    lines = ["=== C2 viability gate: diag (commuting) vs PD-SSM (selective) on non-solvable groups ===",
             f"{'group':>5} {'arm':>5} {'k':>4} {'n':>2} {'med_final':>10} {'med_max':>8} {'grokked':>8}"]
    medf = {}
    for (g, a) in [("S3", "diag"), ("S3", "pd"), ("S5", "diag"), ("S5", "pd")]:
        rs = cell.get((g, a), [])
        if not rs:
            lines.append(f"{g:>5} {a:>5} {'?':>4} {0:>2}   (no runs)")
            continue
        mf = median([r["final_acc"] for r in rs])
        mm = median([r["max_acc"] for r in rs])
        ng = sum(1 for r in rs if r["grok_step"] is not None)
        medf[(g, a)] = mf
        lines.append(f"{g:>5} {a:>5} {rs[0]['k']:>4} {len(rs):>2} {mf:>10.3f} {mm:>8.3f} "
                     f"{f'{ng}/{len(rs)}':>8}")

    def sep(g):
        if (g, "pd") in medf and (g, "diag") in medf:
            return medf[(g, "pd")] - medf[(g, "diag")], medf[(g, "pd")], medf[(g, "diag")]
        return None
    lines.append("")
    s3, s5 = sep("S3"), sep("S5")
    for name, s in [("S3", s3), ("S5", s5)]:
        if s:
            gap, pd, dg = s
            conf = (gap >= 0.40 and pd >= 0.85)
            lines.append(f"{name}: pd={pd:.3f} diag={dg:.3f} gap={gap:+.3f} -> "
                         f"separation {'CONFIRMED' if conf else 'NOT confirmed'}")
    # gate decision keyed on S5 (the non-solvable, depth-robust case)
    if s5:
        gap5, pd5, dg5 = s5
        if pd5 >= 0.85 and gap5 >= 0.40:
            verdict = ("GATE PASS: on the NON-SOLVABLE group S5, PD-SSM (selective) solves "
                       f"(median {pd5:.3f}) while the diagonal (commuting) SSM does not (median {dg5:.3f}, "
                       f"gap {gap5:+.3f}). The expressiveness separation is representational (depth-robust "
                       "per Merrill TC0/NC1), so the Theta(K log K) sample-complexity claim for PD-SSM is "
                       "non-vacuous. Proceed to commit C2 + write the Fano/covering proof sketch.")
        elif dg5 >= 0.85:
            verdict = (f"GATE FAIL / KILL: the diagonal SSM ALSO solves S5 (median {dg5:.3f}). No "
                       "expressiveness gap -> the separation would be an optimization artifact and the "
                       "covering/packing proof is vacuous. Do NOT commit C2; re-hunt.")
        else:
            verdict = (f"GATE INCONCLUSIVE: on S5, pd median {pd5:.3f} (need >=0.85) / diag {dg5:.3f}. "
                       "PD-SSM likely under-budgeted on k=120 -- raise budget/lr before ruling.")
    else:
        verdict = "GATE INCOMPLETE: S5 cells missing."
    lines += ["", "PRE-REGISTERED VERDICT: " + verdict]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "c2_viability.txt"), "w").write(txt + "\n")
    print(f"\n-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
