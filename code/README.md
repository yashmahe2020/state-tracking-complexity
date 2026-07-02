# Complexity of State Tracking in Selective SSMs — code

Reproduction code for the paper. Every reported number traces to a script here and a JSON artifact in
`../results/`. Experiments run on Apple Silicon via MLX; the only dependencies are `mlx` and `numpy`
(`matplotlib` for the figure).

## Setup

```bash
pip install mlx numpy matplotlib
```

All scripts are run from this `code/` directory and write their outputs to `../results/<name>.{json,txt}`.

## Core library

| File | What it provides |
|---|---|
| `groups.py` | Word-problem data for cyclic `Z_k`, symmetric `S_m` (incl. non-abelian `S5`), dihedral `D_n`, and random permutation automata; Cayley-table running products; uniform token sampling. |
| `models.py` | The selective PD-SSM `M(g)=P(g)diag(d(g))` (soft row-stochastic and hard straight-through), the diagonal/commuting SSM baseline, and the GRU baseline. |
| `train.py` | `run_experiment` training loop (optimizer, grad clipping, fresh-test evaluation). |

## Experiments (script → result → paper)

| Script | Result artifact | Paper claim |
|---|---|---|
| `tabular_erm.py` | `tabular_erm.{json,txt}` | T1: tight `Theta(K^2 log K)` ERM rate, optimization-free (slope 2.13/2.14, identical for group and random). |
| `m_star_sweep.py` | `m_star_sweep.{json,txt}`, `m_star_D.{json,txt}` | Statistical `m*` for cyclic `Z_k` and dihedral `D` (the gradient-descent dichotomy, group side). |
| `exp_extended_mstar.py` | `exp_extended_mstar.{json,txt}` | Extended-K (to 48), 5-seed `m*` sweep; normalized-cost curve feeding Figure 1. |
| `exp_gru_baseline.py` | `exp_gru_baseline.{json,txt}` | Non-selective GRU control: same group-vs-random dichotomy (result D is general recurrence, not selectivity-specific). |
| `min_h_sweep.py` | (min-H sweep) | T2 mechanism: minimum width for GD generalization, group sublinear (`~2 sqrt K`) vs random `>= K`. |
| `exp_random_draws.py` | `exp_random_draws.{json,txt}` | Multi-draw random necessity (the `H >= K-1` wall is a class property, not a single-draw artifact). |
| `exp_rand_k16_widths.py` | `exp_rand_k16_widths.{json,txt}` | Random K=16 width/weight-decay sweep: failure is memorization, not capacity (test peaks then falls while train-fit = 1.0). |
| `exp_hard_mstar.py` | `exp_hard_mstar.{json,txt}` | Hard straight-through `m*` spot-check (soft `m*` is not a relaxation artifact). |
| `exp_tsweep.py` | `exp_tsweep.{json,txt}` | T-independence check of the transition-count threshold. |
| `mixing_check.py` | `mixing_check.{json,txt}` | Visited-state marginal is uniform (full-group alphabet => one-step mixing), validating the coverage argument. |
| `c2_viability.py` | `c2_viability.{json,txt}` | S separation viability: PD-SSM tracks non-abelian `S5` exactly (1.000) vs diagonal (0.125). |
| `make_fig_normcost.py` | `../paper/fig_normcost.pdf` | Figure 1 (normalized cost `m*/(K^2 log K)` vs K). |

`c2_proof_sketch.md` is the first-principles derivation underlying the theorems.

## Auxiliary / superseded

`rate_sweep.py` (online optimization-time sweep, superseded by the fixed-dataset `m_star_sweep.py`),
`rank_diag.py` and `probe_deff.py` (participation-ratio diagnostics, superseded by the cleaner min-H
mechanism), `exp_k20_refine.py` (K=20 grid refinement), `precond.py` (preconditioner probe carried over
from the parent project, not load-bearing here).

## Reproducing the headline results

```bash
python3 tabular_erm.py          # T1  (theorem confirmation)
python3 m_star_sweep.py         # D   (group side)
python3 exp_gru_baseline.py     # D   (GRU control)
python3 min_h_sweep.py          # T2  (capacity mechanism)
python3 c2_viability.py         # S   (diagonal separation)
python3 make_fig_normcost.py    # Figure 1
```
