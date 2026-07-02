# Structure, Not Size: The Sample and Parameter Complexity of State Tracking in Selective State-Space Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21126473.svg)](https://doi.org/10.5281/zenodo.21126473)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproduction code for the paper. The work asks what it costs to *learn* finite-state
automata with selective state-space models (the class that includes Mamba), separating the
information-theoretic cost of the problem from what gradient descent actually pays.

**Summary of results.** The sample complexity of empirical risk minimization for tracking a
`K`-state permutation automaton is `Theta(K^2 log K)` transitions, with order-matching bounds,
independent of sequence length and blind to algebraic structure. A parameter-complexity
separation holds for the general linear-transition class: a faithful `O(d_min)`-dimensional
tracker exists for a group, whereas a random `K`-state automaton (`K >= 6`) needs hidden width
at least `K - 1`, which width `K` always attains; a commuting (diagonal) SSM cannot track any
non-abelian automaton at any width or sample size. Empirically, gradient descent is *not*
structure-blind: it generalizes from sub-coverage data on group automata and fails on random
ones despite a perfectly fit training set, for both a selective SSM and a non-selective GRU.

## Layout

```
code/      model implementations, automaton generators, training loop, and all experiment scripts
results/   the JSON + TXT artifacts every reported table and figure is computed from
```

Every number in the paper traces to a script in `code/` and a result artifact in `results/`.
The full script -> result -> paper-claim map is in [`code/README.md`](code/README.md).

## Setup

```bash
pip install -r code/requirements.txt   # mlx==0.31.2, numpy==1.26.4, matplotlib==3.8.0
```

Experiments run on Apple Silicon via MLX. All scripts are run from the `code/` directory and
write outputs to `../results/`.

## Reproducing the headline results

```bash
cd code
python3 tabular_erm.py          # tight Theta(K^2 log K) ERM rate, optimization-free
python3 m_star_sweep.py         # gradient-descent dichotomy, group side
python3 exp_gru_baseline.py     # non-selective GRU control (effect is general recurrence)
python3 min_h_sweep.py          # capacity mechanism: group sublinear vs random >= K
python3 c2_viability.py         # diagonal separation: PD-SSM tracks S5 exactly, diagonal cannot
python3 make_fig_normcost.py    # Figure 1 (normalized cost m*/(K^2 log K) vs K)
```

## Citation

If you use this code, please cite the paper and the archived software release (see
`CITATION.cff` and the Zenodo DOI above).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The accompanying paper is
licensed CC BY 4.0.
