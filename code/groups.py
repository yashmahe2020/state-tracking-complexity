"""Group word-problem state-tracking tasks.

Each group G is given by its Cayley (multiplication) table. The state-tracking task:
  input  x_1 .. x_T   : a sequence of group elements (tokens in {0..|G|-1}), sampled uniformly
  target y_1 .. y_T   : the running product  y_t = x_1 * x_2 * ... * x_t  in G

The model must implicitly learn the Cayley table and compose it left-to-right. The number of
distinct transitions (input symbols) equals |G| = k, and the number of automaton states is |G|.

Groups implemented:
  Z_n  (n = 2,4,8,16) : cyclic, abelian; a*b = (a+b) mod n.   REPRESENTABLE by a complex-diagonal
                        SSM (n-th roots of unity), so any learning failure is a LEARNABILITY result.
  S_3                 : symmetric group on 3 symbols, 6 elements, NON-abelian (transitions do not
                        commute) -> not representable by a purely commuting diagonal transition.

The identity element is always index 0.
"""
from __future__ import annotations

import itertools
import numpy as np


def cyclic_table(n: int) -> np.ndarray:
    """Cayley table of Z_n with element i representing the residue i. Identity = 0."""
    idx = np.arange(n)
    return (idx[:, None] + idx[None, :]) % n


def symmetric_table(m: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Cayley table of S_m (symmetric group on m symbols), order m!.

    Elements are all permutations of (0..m-1) as tuples; element*other means "apply `other`
    first, then `self`" (function composition), i.e. (p*q)[i] = p[q[i]]. Identity = (0,1,..,m-1)
    which is placed at index 0.
    """
    perms = list(itertools.permutations(range(m)))
    # ensure identity is index 0
    identity = tuple(range(m))
    perms.remove(identity)
    perms = [identity] + perms
    index = {p: i for i, p in enumerate(perms)}
    k = len(perms)
    table = np.zeros((k, k), dtype=np.int64)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            comp = tuple(p[q[s]] for s in range(m))  # (p*q)[s] = p[q[s]]
            table[i, j] = index[comp]
    return table, perms


def dihedral_table(n: int) -> np.ndarray:
    """Cayley table of the dihedral group D_n, order 2n, NON-abelian for n>=3.

    Elements are (f, a) with f in {0,1} (reflection flag) and a in Z_n (rotation), indexed
    idx = f*n + a; identity = (0,0) = index 0. Group law (s^f r^a convention, r^a s = s r^{-a}):
        (f1,a1) * (f2,a2) = (f1 xor f2, ((-1)^{f1} * a2 + a1) mod n).
    D_n has a FAITHFUL 2-D real representation (symmetries of the n-gon), so its effective
    representation dimension d_eff = 2 (vs 1 for cyclic Z_n) -- used as the d_eff=2 rung of the
    rate-spectrum ladder. NON-abelian -> not representable by a commuting diagonal SSM.
    """
    k = 2 * n
    table = np.zeros((k, k), dtype=np.int64)
    for f1 in (0, 1):
        for a1 in range(n):
            i = f1 * n + a1
            for f2 in (0, 1):
                for a2 in range(n):
                    j = f2 * n + a2
                    f = f1 ^ f2
                    a = ((-1) ** f1 * a2 + a1) % n
                    table[i, j] = f * n + a
    return table


def random_perm_table(k: int, seed: int) -> np.ndarray:
    """Transition table of a RANDOM permutation automaton: K states, K input symbols, where each
    input symbol g acts as an independent uniformly-random PERMUTATION of the K states.

    table[s, g] = perm_g[s]. Symbol 0 is fixed to the identity permutation so the start state
    (index 0) and an identity transition exist (parity with the group case). The K permutations
    carry NO algebraic relation, so the transition monoid they generate is generically large and
    high-dimensional (d_eff ~ K): there is no cross-symbol or cross-state shortcut, so a learner
    must observe every (state, symbol) cell -> the coupon-collector bound Theta(K^2 log K) is TIGHT.
    This is the high-d_eff endpoint of the rate-spectrum ladder and the same hypothesis class
    (input-selected permutations) as the PD-SSM, with the algebraic structure removed.
    """
    rng = np.random.default_rng(10_000 + seed)
    table = np.zeros((k, k), dtype=np.int64)
    table[:, 0] = np.arange(k)                       # symbol 0 = identity
    for g in range(1, k):
        table[:, g] = rng.permutation(k)
    return table


GROUP_SPECS = {
    "Z2": ("cyclic", 2),
    "Z4": ("cyclic", 4),
    "Z8": ("cyclic", 8),
    "Z16": ("cyclic", 16),
    "S3": ("symmetric", 3),
}


def get_group(name: str) -> tuple[np.ndarray, int]:
    """Return (cayley_table [k,k], order k) for a named group.

    Accepts the registered names in GROUP_SPECS, plus a generic 'Z<n>' for any cyclic order n
    (e.g. 'Z12', 'Z20') so the confirmatory critical-k sweep can use arbitrary cyclic groups.
    """
    if name in GROUP_SPECS:
        kind, param = GROUP_SPECS[name]
        if kind == "cyclic":
            return cyclic_table(param), param
        elif kind == "symmetric":
            t, _ = symmetric_table(param)
            return t, t.shape[0]
    if name.startswith("Z") and name[1:].isdigit():
        n = int(name[1:])
        return cyclic_table(n), n
    if name.startswith("S") and name[1:].isdigit():
        m = int(name[1:])
        t, _ = symmetric_table(m)            # S_m, order m! (S5 -> k=120, NON-SOLVABLE)
        return t, t.shape[0]
    if name.startswith("D") and name[1:].isdigit():
        n = int(name[1:])                    # dihedral D_n, order 2n, d_eff=2, non-abelian (n>=3)
        return dihedral_table(n), 2 * n
    if name.startswith("RAND"):
        # RAND<k> or RAND<k>s<seed>: random permutation automaton, K states/symbols, d_eff ~ K
        rest = name[4:]
        if "s" in rest:
            ks, sd = rest.split("s"); k, seed = int(ks), int(sd)
        else:
            k, seed = int(rest), 0
        return random_perm_table(k, seed), k
    raise ValueError(name)


def running_product(seq: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Left-to-right running product of a token sequence under a Cayley table.

    seq: [..., T] int tokens. Returns [..., T] of running-product states.
    state_t = state_{t-1} * x_t  with state_0 = identity (index 0).
    """
    *batch, T = seq.shape
    out = np.empty_like(seq)
    state = np.zeros(batch, dtype=seq.dtype)  # identity = 0
    for t in range(T):
        state = table[state, seq[..., t]]
        out[..., t] = state
    return out


def make_batch(
    group: str,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a batch of (inputs, targets) for a group word problem.

    inputs  : [B, T] int tokens (group elements), uniform over the k symbols.
    targets : [B, T] int running-product states.

    stratified=True guarantees every one of the k symbols appears at least once somewhere in the
    batch (confound Row 9: avoids an undefined Gram row for an unobserved transition). It does NOT
    change the per-position marginal materially for our B*T >> k regime; we simply reseed any batch
    that misses a symbol.
    """
    table, k = get_group(group)
    while True:
        x = rng.integers(0, k, size=(batch_size, seq_len), dtype=np.int64)
        if not stratified or np.unique(x).size == k:
            break
    y = running_product(x, table)
    return x, y
