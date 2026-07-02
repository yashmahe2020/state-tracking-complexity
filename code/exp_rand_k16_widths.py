"""Archival run (Phase-4 JMLR audit, provenance fix): the paper's random-failure paragraph states the
single random K=16 automaton fails to generalize at hidden width up to 128 (= 8K), fitting its training
set (0.98 to 1.00) while reaching only fresh-test sequence accuracy 0.10 to 0.33, under weight decay
0.01 to 0.1 and up to 40000 steps. That claim previously rested on an inline smoke run with no archived
JSON artifact (a provenance ORPHAN). This script reproduces it as a reproducible artifact.

Protocol matches exp_random_draws.py for draw 0 (group RAND16s0, train rng = 1000+0, test rng = 7e6+0,
model key = 0), so the H=32 (=2K) cell reproduces the existing exp_random_draws result and H=64, H=128
extend the same single draw to 4K and 8K. For each (H, wd) we report best fresh-test sequence accuracy
AND final train-set sequence accuracy (on 512 of the fixed 3000 training sequences). Idempotent per
(H, wd). Writes results/exp_rand_k16_widths.{json,txt}. NO claim depends on a specific value; whatever
the run measures is what the paper will report.
"""
from __future__ import annotations
import os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "exp_rand_k16_widths.json")

K = 16
DRAW = 0
GROUP = "RAND16s0"
H_GRID = [32, 64, 128]          # 2K, 4K, 8K
WD_GRID = [0.0, 0.01, 0.1]
T = 16
THR = 0.95
STEPS = 40000
EVAL_EVERY = 2000
LR = 0.01
N_SEQ = 3000
EVAL_FRESH = 512
MAX_WORKERS = 3


def worker(task):
    H, wd = task
    sys.path.insert(0, HERE)
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from groups import make_batch
    from models import pd_init, pd_forward, loss_fn
    dr = np.random.default_rng(1000 + DRAW)
    Xtr, Ytr = make_batch(GROUP, N_SEQ, T, dr, stratified=False)
    Xtr, Ytr = mx.array(Xtr), mx.array(Ytr)
    te = np.random.default_rng(7_000_000 + DRAW)
    Xte, Yte = make_batch(GROUP, EVAL_FRESH, T, te, stratified=False)
    Xte, Yte = mx.array(Xte), mx.array(Yte)
    Xtr_eval, Ytr_eval = Xtr[:EVAL_FRESH], Ytr[:EVAL_FRESH]
    p = pd_init(K, H, K, mx.random.key(DRAW))
    opt = (optim.AdamW(learning_rate=LR, weight_decay=wd) if wd > 0
           else optim.Adam(learning_rate=LR))
    lg = mx.value_and_grad(lambda pp, xx, yy: loss_fn(pd_forward, pp, xx, yy))
    rng = np.random.default_rng(DRAW)
    best_test = 0.0
    train_acc = 0.0
    for s in range(1, STEPS + 1):
        idx = rng.integers(0, N_SEQ, size=64)
        _, g = lg(p, Xtr[mx.array(idx)], Ytr[mx.array(idx)])
        opt.update(p, g); mx.eval(p, opt.state)
        if s % EVAL_EVERY == 0:
            test = float(mx.mean(mx.all(mx.argmax(pd_forward(p, Xte), axis=-1) == Yte, axis=1).astype(mx.float32)))
            best_test = max(best_test, test)
            train_acc = float(mx.mean(mx.all(mx.argmax(pd_forward(p, Xtr_eval), axis=-1) == Ytr_eval, axis=1).astype(mx.float32)))
    return {"k": K, "draw": DRAW, "H": H, "H_over_K": H // K, "wd": wd,
            "train_seq_acc": train_acc, "best_test_seq_acc": best_test,
            "generalizes": best_test >= THR}


def main():
    os.makedirs(RES, exist_ok=True)
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(r["H"], r["wd"]) for r in done}
    tasks = [(H, wd) for H in H_GRID for wd in WD_GRID if (H, wd) not in seen]
    print(f"RAND16 width archival: {len(tasks)} runs, {MAX_WORKERS} workers", flush=True)
    results = list(done)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print(f"  FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True); continue
            results.append(r); json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [{i}/{len(tasks)}] H={r['H']:<4}(={r['H_over_K']}K) wd={r['wd']:<4} "
                  f"train={r['train_seq_acc']:.2f} test={r['best_test_seq_acc']:.2f} "
                  f"{'GEN' if r['generalizes'] else '.'}", flush=True)

    tests = [r["best_test_seq_acc"] for r in results]
    trains = [r["train_seq_acc"] for r in results]
    lines = ["=== RAND K=16 single draw (RAND16s0): width sweep 2K, 4K, 8K x wd {0,0.01,0.1} ===",
             f"N_SEQ={N_SEQ} ({N_SEQ*T} transitions), steps={STEPS}, fresh-test bar {THR}",
             f"{'H':>5} {'H/K':>4} {'wd':>5} {'train':>7} {'test':>7} {'gen':>4}"]
    for H in H_GRID:
        for wd in WD_GRID:
            cell = [r for r in results if r["H"] == H and r["wd"] == wd]
            if not cell:
                continue
            r = cell[0]
            lines.append(f"{H:>5} {H//K:>4} {wd:>5} {r['train_seq_acc']:>7.2f} "
                         f"{r['best_test_seq_acc']:>7.2f} {'GEN' if r['generalizes'] else '.':>4}")
    if tests:
        lines += ["",
                  f"train-fit range: {min(trains):.2f} to {max(trains):.2f}",
                  f"fresh-test range: {min(tests):.2f} to {max(tests):.2f}  (bar {THR})",
                  f"VERDICT: {'ALL FAIL to generalize (test < bar) at every width up to 8K' if max(tests) < THR else 'SOME GENERALIZE -> inspect'}"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "exp_rand_k16_widths.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
