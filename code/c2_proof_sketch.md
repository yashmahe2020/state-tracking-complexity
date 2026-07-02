# Theory for the selective-SSM learnability paper (final structure, advisor ruling #3)

The paper makes FOUR claims. Two are theorems (wall-free, standard tools), one is an empirical
dichotomy (with the hard direction's lower bound parked as Future Work, per advisor), one is a
qualitative separation. Every claim is matched by an experiment in `results/`.

  T1  ERM sample complexity = Theta(K^2 log K) for any K-state permutation automaton (tight). [theorem]
  D   GD dichotomy: GD-trained PD-SSMs BEAT the ERM rate on GROUP automata (empirical K^1.6) but FAIL
      to generalize on RANDOM automata at any feasible sample size (memorize). [empirical]
  T2  Capacity mechanism (wall-free): a faithful O(d_min)-dim PD-SSM solution EXISTS for a group
      (d_min = O(1) cyclic/dihedral, O(sqrt K) symmetric); any exact tracker for a RANDOM K-automaton
      needs hidden width H >= K. [theorem] Empirically GD realizes the group solution at H ~ 2 sqrt K
      (<< K, sublinear) while random is GD-unlearnable at H <= 2K for K >= 12.
  S   Diagonal (commuting) SSMs cannot track any non-commuting automaton at any width or sample size.

The story: ERM needs full coverage (T1), so structure cannot help a structure-blind learner; but a
GD-trained selective SSM DOES exploit structure (D), and T2 explains why -- a low-dimensional faithful
solution exists for groups and not for random automata. The gap between the ERM curve (slope 2) and the
GD-on-groups curve (slope 1.6), and the min-H capacity separation, are the two mechanism figures.

## Setup (the learning problem)

- Input alphabet Sigma, |Sigma| = K symbols. State set Q, |Q| = K. Target transition delta: Q x Sigma
  -> Q with delta(., g) a permutation of Q for each g (a deterministic, invertible automaton). A GROUP
  word problem is the structured case delta(s,g) = s*g; a RANDOM automaton draws the K permutations
  {M_g} i.i.d. uniform from Sym(Q) (no algebraic relations). Realizable class H = K-tuples of
  permutations, |H| = (K!)^K, log|H| = Theta(K^2 log K) bits.
- PD-SSM realizes this class: state in R^H, symbol g applies M(g) = P(g) diag(d(g)) (P row-stochastic,
  the hard limit a permutation); linear readout R^H -> R^K decodes the state. H is the hidden width
  (a model knob, NOT necessarily K).
- Data: sequences x_1..x_T, x_t ~ Unif(Sigma) i.i.d., labeled by running states; one length-T sequence
  = T labeled transition observations (s_{t-1}, x_t) -> s_t. Sample complexity counted in TRANSITIONS m
  (= m/T sequences); this is the T-independent quantity. Full-symbol alphabet => the visited-state
  marginal is exactly uniform for t>=1 in any finite group (one-step mixing; verified Z16 maxdev 3.6e-4,
  S5 1.5e-4 over 200k T=16 seqs, mixing_check.py), so cells (s,g) are hit uniformly over the K^2 universe.
- Exact-tracking criterion: predict the whole running product w.h.p. One wrong transition corrupts the
  suffix, so sequence accuracy demands learning delta on (essentially) all K^2 cells.

## T1. ERM sample complexity is Theta(K^2 log K) (tight). [matches tabular_erm: slope 2.13/2.14, CV 0.14]

UPPER O(K^2 log K). Each observation reveals one of K^2 cells, hit uniformly => coupon collector over
N = K^2 coupons. m = c K^2 log K leaves a cell unseen with prob <= K^2 (1-1/K^2)^m <= K^2 e^{-c log K}
-> 0 for c > 2. Once all cells are seen, the only K-tuple of permutations consistent with the data is
delta itself (realizable ERM) => exact recovery, zero test error for all T. Permutation closure (last
cell of a column forced once K-1 are seen) tightens the constant, not the order.

LOWER Omega(K^2 log K). Coverage necessity: if cell (s,g) is unobserved, two hypotheses in H agree with
all data but disagree on (s,g) (swap two unobserved targets within column g -- both valid permutations),
so the learner cannot beat chance there; meeting (s,g) in a length-T test sequence (prob ~1/K^2 per
step) gives a wrong suffix. Vanishing error thus requires all-but-o(1) cells observed; the number unseen
after m draws is ~K^2 e^{-m/K^2}, which exceeds 1 unless m >= K^2 ln K^2 = Theta(K^2 log K). Fano
cross-check: log|H| = Theta(K^2 log K) bits, <= log K bits per transition => Omega(K^2) crude; the extra
log K is the coupon coverage term, binding under exact tracking.

CRITICAL POINT for the paper: T1 holds for ANY consistent learner and is BLIND to structure -- it is the
SAME Theta(K^2 log K) for group and random targets, because a structure-blind learner must cover all
cells either way. This is exactly what the tabular learner shows (Z and RAND IDENTICAL, slope 2.13 vs
2.14). So T1 is the architecture-independent baseline; structure can only help a learner whose inductive
bias exploits it -- which is the GD result D, not ERM.

T-independence (Wall A closed): permutation transition matrices have operator norm exactly 1, so the
hidden state is unit-norm at every step and the recurrence covering number accumulates NO e^{O(T)}
norm-product blow-up. The threshold is on the transition count, independent of T.

## D. The GD dichotomy (empirical). [matches m_star_sweep (Z), Phase-2C random runs]

On GROUP automata, GD-trained PD-SSMs generalize at SUB-ERM sample size: measured statistical m* scales
as K^1.62 (r^2 0.91), strictly below the ERM Theta(K^2 log K) (slope 2.13). GD's implicit bias exploits
the algebraic structure (a generator's action is consistent across states), generalizing to cells it has
not covered. On RANDOM automata, GD does NOT generalize at any feasible budget: at K=16, with n = 31x
the ERM coverage m*, H up to 128, up to 40000 steps, and weight decay 0.01-0.1, fresh-test seq-acc stays
0.10-0.33 while train-fit is ~1.0 (pure memorization); Z16 at the same budget reaches 1.00.

Honest scope (advisor): the SUB-ERM group rate is an empirical OBSERVATION; its matching lower bound
(prove GD needs >= K^1.x) requires controlling GD's trajectory, which is the optimization-to-statistics
gap PARKED as Future Work (it is the same wall that capped the parent project). We do not claim it as a
theorem. What IS a theorem is why a sub-ERM solution can exist at all -- T2.

## T2. Capacity mechanism (wall-free, hypothesis-class). [matches min_h_sweep, bf5aw8d59 pre-check]

The reason GD can beat ERM on groups but not on random is that a LOW-DIMENSIONAL faithful solution
EXISTS for groups and not for random automata -- a statement about the hypothesis class, no GD needed.

(a) EXISTENCE for groups. A finite group of order K has a faithful representation of dimension d_min:
d_min = 2 for cyclic Z_k and dihedral D_n (the rotation/reflection action on R^2), and for the symmetric
group S_m (order K = m!) the minimal faithful rep is the (m-1)-dim standard rep, so d_min = m-1; by
Stirling log K = m log m (1+o(1)), hence m = Theta(log K / log log K) and d_min = Theta(log K / log log K)
(NOT O(sqrt K), which is loose by a superpolynomial factor; verified m=8: d_min=7 vs sqrt(K)=201).
Embedding Q into R^{d_min} by the representation and reading out the K elements as their d_min-dim images
(distinct points -- e.g. K-th roots of unity for Z_k) gives an EXACT tracker of width O(d_min) << K.
(Caveat realized in experiment: the specific P diag(d) parameterization with P row-stochastic cannot
encode a pure 2-D rotation -- column signs are tied to d -- so GD does not reach H=d_min=2; it reaches a
faithful solution at H ~ 2 sqrt K instead. The existence claim stands for the hypothesis class; the
achievable-by-GD width is the empirical min-H below.)

(b) NECESSITY for random. For a random K-automaton the generated transition monoid acts faithfully and
(generically) irreducibly on R^K: the K i.i.d. random permutations generate, with high probability, a
group that is 2-transitive (in fact A_K or S_K), whose only invariant subspaces of the permutation
representation are the trivial line and its K-1 dim complement. Tracking requires distinguishing all K
states under operators with no common lower-dim invariant subspace beyond that split, so any exact
tracker needs hidden width H >= K-1. There is no low-dim solution to find.

EMPIRICAL min-H (the capacity-separation figure): smallest H with GD fresh-test seq-acc >= 0.95.
GROUP Z_k: min-H = 6, 8, 8, 10, 8 for K = 8,12,16,20,24 -- minH/K falls 0.75 -> 0.33, minH/sqrt(K) ~ 2.0
(roughly constant ~8); log-log slope +0.33, SUBLINEAR. RANDOM: min-H = 16 (= 2K) at K=8, then
GD-UNLEARNABLE at H <= 2K for K >= 12. So group capacity grows sublinearly (a low-dim solution exists and
GD finds it) while random capacity is >= K and diverges (no low-dim solution exists). This is the
wall-free mechanism that explains D.

## S. Separation from the diagonal (commuting) SSM. [matches c2_viability: S5 PD 1.000 vs diag 0.125]

Diagonal / commuting transitions realize only ABELIAN composition, so a diagonal SSM cannot represent a
non-abelian (a fortiori non-solvable) automaton at ANY width or sample size (Merrill et al. "Illusion of
State" TC0-vs-NC1; diagonal-SSM expressivity limits). On non-solvable S5 (K=120) the diagonal class has
NO realizable hypothesis -> infinite sample complexity; PD-SSM tracks it exactly (1.000 vs 0.125, gap
+0.875, depth-robust). The separation is qualitative and unbounded (finite vs infinite), not a rate gap.
Note this is the GROUP-STRUCTURE-vs-RANDOM axis crossed with the COMMUTING-vs-SELECTIVE axis: PD-SSM
learns non-abelian groups (S5) fine; what it cannot learn is STRUCTURELESS (random) automata. So the two
divisions are distinct: (commuting => abelian-only, S) and (selective+GD => structured-only, D/T2).

## Headline theorem (paper statement)

THEOREM (informal). Tracking a K-state permutation automaton needs Theta(K^2 log K) transitions under
ERM (tight, any consistent learner) -- structure-blind. A GD-trained selective (PD-SSM) recurrence beats
this on algebraically structured (group) automata, generalizing from sub-coverage data, because a
faithful O(d_min)-dimensional solution exists in the hypothesis class (d_min = O(1) for cyclic/dihedral,
O(sqrt K) for symmetric) and is reached at hidden width H ~ 2 sqrt K << K; for random K-state automata
no exact tracker exists below width K, and GD fails to generalize at any sample size. A commuting
(diagonal) SSM cannot track any non-commuting automaton at all. Selective recurrences are thus
structure-adaptive learners; their practical sample- and parameter-efficiency is governed by the
algebraic structure of the target, not its raw state count.

FALSIFIABLE PREDICTIONS: (1) ERM / tabular slope 2 in log K (CONFIRMED 2.13/2.14). (2) GD-on-groups slope
< 2 (CONFIRMED 1.62). (3) group min-H sublinear in K, random min-H >= K (CONFIRMED: +0.33 vs unlearnable).
(4) diagonal infinite-vs-PD-finite on non-solvable groups (CONFIRMED S5).

OPEN / FUTURE WORK: the matching lower bound for the GD-on-groups sub-ERM rate (optimization-to-
statistics gap); non-uniform / adaptive sampling; deep selective stacks; closing the d_min-vs-2sqrtK gap
between the existence bound and the GD-achievable width (a parameterization question).
