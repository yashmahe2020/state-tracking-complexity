"""Sequence models for group word-problem state tracking, in MLX.

Two architectures, sharing an identical linear readout head:

  DiagonalSSM (Arm A) - input-selected COMPLEX-DIAGONAL transition, NORM-PRESERVING (unitary). The
    recurrent state is d_c complex channels, realised as 2*d_c real dims (real, imag). Token g selects
    a per-channel UNITARY factor  a_g[c] = exp(i * theta_g[c])  (magnitude exactly 1). Fixing the
    magnitude to 1 removes geometric state decay, which would otherwise make EVERY arm fail on long
    sequences from forgetting alone (a confound); group word problems are intrinsically unitary
    (Z_n = rotation by 2*pi/n). theta is unconstrained so NEGATIVE real eigenvalues (theta=pi) remain
    reachable (confound Row 5; Grazzi/Siems: negative eigenvalues matter for state tracking).
    Transition-only dynamics  h_t = A(x_t) h_{t-1}, learned h_0 ; the input enters through WHICH
    diagonal transition is applied. The single transition-parameter family {omega} ([k, d_c]) is the
    object whose per-generator gradient rows form the Gram matrix G.

  PDSSM (Arm C) - input-selected  M(g) = P(g) @ diag(d(g))  with P(g) a row-softmax soft permutation
    (learned [d,d] logits per generator) and d(g) in [-1,1]. Non-commuting -> FSA-complete; serves
    as a solvability ceiling (NOT parameter-matched; confound Row 4). Real state of width H = 2*d_c.

Both produce logits [B, T, n_classes] at every position. Forward is an explicit O(T) scan; T<=256
and the tensors are tiny, so this is cheap.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn


# ----------------------------- Arm A: Diagonal SSM -----------------------------

def diag_init(k: int, d_c: int, n_classes: int, key) -> dict:
    """k = #generators (= group order), d_c = #complex channels (H = 2*d_c real state dims).

    Unitary diagonal: magnitude is fixed to 1, so the only transition parameter is the angle omega.
    """
    ks = mx.random.split(key, 4)
    return {
        # angles: broad random init covers the roots of unity needed for cyclic groups.
        "omega": math.pi * mx.random.normal((k, d_c), key=ks[0]),
        "h0r": 0.1 * mx.random.normal((d_c,), key=ks[1]),
        "h0i": 0.1 * mx.random.normal((d_c,), key=ks[2]),
        "W": (1.0 / math.sqrt(2 * d_c)) * mx.random.normal((n_classes, 2 * d_c), key=ks[3]),
        "b": mx.zeros((n_classes,)),
    }


def diag_forward(p: dict, x: mx.array) -> mx.array:
    """x: [B,T] int tokens. Returns logits [B,T,n_classes]."""
    B, T = x.shape
    d_c = p["omega"].shape[1]
    c_all = mx.cos(p["omega"])            # real part of unit-modulus factor
    s_all = mx.sin(p["omega"])            # imag part
    hr = mx.broadcast_to(p["h0r"], (B, d_c))
    hi = mx.broadcast_to(p["h0i"], (B, d_c))
    outs = []
    for t in range(T):
        g = x[:, t]                        # [B]
        c = c_all[g]                       # [B,d_c]
        s = s_all[g]
        nhr = hr * c - hi * s
        nhi = hr * s + hi * c
        hr, hi = nhr, nhi
        h = mx.concatenate([hr, hi], axis=-1)      # [B,2d_c]
        outs.append(h @ p["W"].T + p["b"])         # [B,n_classes]
    return mx.stack(outs, axis=1)          # [B,T,n_classes]


def diag_forward_fast(p: dict, x: mx.array) -> mx.array:
    """Vectorized equivalent of diag_forward (NO Python T-loop): ~100x faster for long T.

    Unit-modulus diagonal SSM => R(a)R(b)=R(a+b), so the state after t tokens is h0 rotated by the
    CUMULATIVE phase. Output at position t uses the state AFTER applying token x[:,t] (matches the loop).
    Machine-verified identical to diag_forward by code/verify_fast_forward.py before any use.
    """
    omega = p["omega"]                        # [k, d_c]
    ang = omega[x]                            # gather by token -> [B, T, d_c]
    cum = mx.cumsum(ang, axis=1)              # cumulative phase after token t -> [B, T, d_c]
    cc = mx.cos(cum)
    ss = mx.sin(cum)
    hr = p["h0r"] * cc - p["h0i"] * ss        # R(cum) applied to h0 -> [B, T, d_c]
    hi = p["h0r"] * ss + p["h0i"] * cc
    h = mx.concatenate([hr, hi], axis=-1)     # [B, T, 2 d_c]
    return h @ p["W"].T + p["b"]              # [B, T, n_classes]


TRANSITION_KEYS = ("omega",)  # the parameter(s) the Gram preconditioner acts on (Arm A, unitary)


# ----------------------------- Arm C: PD-SSM ceiling -----------------------------

def pd_init(k: int, H: int, n_classes: int, key) -> dict:
    ks = mx.random.split(key, 5)
    # P logits init near identity permutation: large diagonal so softmax(row) ~ one-hot self
    eye_bias = 3.0 * mx.broadcast_to(mx.eye(H), (k, H, H))
    return {
        "P_logits": eye_bias + 0.1 * mx.random.normal((k, H, H), key=ks[0]),
        "d_raw": mx.random.normal((k, H), key=ks[1]) * 0.1,   # d = tanh(d_raw) in (-1,1)
        "h0": 0.1 * mx.random.normal((H,), key=ks[2]),
        "W": (1.0 / math.sqrt(H)) * mx.random.normal((n_classes, H), key=ks[3]),
        "b": mx.zeros((n_classes,)),
    }


def pd_forward(p: dict, x: mx.array) -> mx.array:
    B, T = x.shape
    H = p["d_raw"].shape[1]
    P = mx.softmax(p["P_logits"], axis=-1)         # [k,H,H] row-stochastic
    d = mx.tanh(p["d_raw"])                          # [k,H] in (-1,1)
    M = P * d[:, None, :]                            # [k,H,H]  = P @ diag(d)
    h = mx.broadcast_to(p["h0"], (B, H))
    outs = []
    for t in range(T):
        g = x[:, t]                                 # [B]
        Mg = M[g]                                    # [B,H,H]
        h = mx.einsum("bij,bj->bi", Mg, h)          # [B,H]
        outs.append(h @ p["W"].T + p["b"])
    return mx.stack(outs, axis=1)


# ------------------- Arm C-hard: PD-SSM with hard (STE) permutation -------------------
# Realizes the THEORY's hypothesis class: M(g) is a HARD permutation matrix (norm 1, exact FSA
# transition). Forward uses a hard per-row argmax of P_logits (a 0/1 row-stochastic matrix; a
# permutation once the rows pick distinct columns); backward uses the soft-softmax gradient
# (straight-through estimator). No eigenvalue scaling d (pure permutation): norm-preserving, so the
# state does not decay and the model LENGTH-GENERALIZES once it has learned the correct permutations
# (unlike the soft Arm C, which fits a fixed training length but fails OOD). h0 is learned dense.

def pd_hard_init(k: int, H: int, n_classes: int, key, eye_bias: float = 1.0) -> dict:
    ks = mx.random.split(key, 4)
    # eye_bias 1.0 (was 3.0): a sharp identity prior creates dead STE basins (k>=16 seeds get stuck);
    # 1.0 keeps the permutation prior while leaving the softmax gradient informative (diagnostic-verified).
    eb = eye_bias * mx.broadcast_to(mx.eye(H), (k, H, H))
    return {
        "P_logits": eb + 0.1 * mx.random.normal((k, H, H), key=ks[0]),
        "h0": 0.1 * mx.random.normal((H,), key=ks[1]),
        "W": (1.0 / math.sqrt(H)) * mx.random.normal((n_classes, H), key=ks[2]),
        "b": mx.zeros((n_classes,)),
    }


def _ste_hard_perm(P_logits: mx.array, beta: float = 1.0) -> mx.array:
    """Row-wise straight-through hard permutation with a TEMPERATURE-annealed soft backward.
    forward = one-hot(argmax) per row (a hard 0/1 permutation, norm 1 -> length-generalizes);
    backward = gradient of softmax(beta * P_logits). beta is the inverse temperature: small beta ->
    smooth, wide-support gradient (escapes dead basins early); large beta -> sharp (matches the hard
    forward late). Annealing beta low->high over training fixes the STE optimization fragility while
    keeping the forward EXACTLY a permutation. P_logits: [k,H,H] -> [k,H,H]."""
    soft = mx.softmax(beta * P_logits, axis=-1)
    H = P_logits.shape[-1]
    idx = mx.argmax(P_logits, axis=-1)                       # [k,H]
    hard = (mx.arange(H) == idx[..., None]).astype(soft.dtype)  # one-hot rows [k,H,H]
    return mx.stop_gradient(hard - soft) + soft             # forward hard, grad through soft(beta)


def pd_hard_forward(p: dict, x: mx.array, beta: float = 1.0) -> mx.array:
    B, T = x.shape
    H = p["h0"].shape[0]
    M = _ste_hard_perm(p["P_logits"], beta)                 # [k,H,H] hard forward, soft(beta) backward
    h = mx.broadcast_to(p["h0"], (B, H))
    outs = []
    for t in range(T):
        Mg = M[x[:, t]]                                     # [B,H,H]
        h = mx.einsum("bij,bj->bi", Mg, h)                 # [B,H]
        outs.append(h @ p["W"].T + p["b"])
    return mx.stack(outs, axis=1)


# ------------------- Baseline: non-selective GRU (Phase-4 ruling #5, EXP-6) -------------------
# A standard 1-layer GRU. Unlike the PD-SSM, the input does NOT select a transition matrix: the
# token enters only additively through a learned embedding, while the recurrent transition (the GRU
# cell weights U_*) is the SAME for every input. This is the non-selective recurrent control that
# isolates whether the group/random dichotomy is a property of the SELECTIVE inductive bias or of
# any gradient-trained recurrent model. Same dict-param + forward(p, x) -> [B,T,C] convention, so it
# plugs into loss_fn / accuracy and the m* sweep harness unchanged. Hidden width H is the knob.

def gru_init(k: int, H: int, n_classes: int, key, emb: int | None = None) -> dict:
    if emb is None:
        emb = H
    ks = mx.random.split(key, 9)
    s_in = 1.0 / math.sqrt(emb)
    s_h = 1.0 / math.sqrt(H)
    return {
        "E":  0.1 * mx.random.normal((k, emb), key=ks[0]),       # token embedding (additive input)
        "Wz": s_in * mx.random.normal((H, emb), key=ks[1]), "Uz": s_h * mx.random.normal((H, H), key=ks[2]),
        "Wr": s_in * mx.random.normal((H, emb), key=ks[3]), "Ur": s_h * mx.random.normal((H, H), key=ks[4]),
        "Wn": s_in * mx.random.normal((H, emb), key=ks[5]), "Un": s_h * mx.random.normal((H, H), key=ks[6]),
        "bz": mx.zeros((H,)), "br": mx.zeros((H,)), "bn": mx.zeros((H,)),
        "h0": 0.1 * mx.random.normal((H,), key=ks[7]),
        "W":  s_h * mx.random.normal((n_classes, H), key=ks[8]), "b": mx.zeros((n_classes,)),
    }


def gru_forward(p: dict, x: mx.array) -> mx.array:
    B, T = x.shape
    H = p["h0"].shape[0]
    e_all = p["E"][x]                                            # [B,T,emb]
    h = mx.broadcast_to(p["h0"], (B, H))
    outs = []
    for t in range(T):
        e = e_all[:, t]                                         # [B,emb]
        z = mx.sigmoid(e @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
        r = mx.sigmoid(e @ p["Wr"].T + h @ p["Ur"].T + p["br"])
        n = mx.tanh(e @ p["Wn"].T + r * (h @ p["Un"].T) + p["bn"])
        h = (1.0 - z) * n + z * h
        outs.append(h @ p["W"].T + p["b"])
    return mx.stack(outs, axis=1)


# ------------- Baselines for UNIVERSALITY: vanilla RNN + LSTM (non-selective) -------------
# Same non-selective convention as the GRU: the token enters only additively through a learned
# embedding; the recurrent transition is the SAME for every input. These extend the dichotomy's
# control set from {GRU} to {GRU, vanilla-RNN, LSTM}, testing whether group=generalize / random=fail
# is universal across gradient-trained recurrent models rather than a GRU quirk. Identical
# dict-param + forward(p,x)->[B,T,C] convention so they drop into loss_fn / the m* sweep unchanged.

def rnn_init(k: int, H: int, n_classes: int, key, emb: int | None = None) -> dict:
    if emb is None:
        emb = H
    ks = mx.random.split(key, 5)
    s_in = 1.0 / math.sqrt(emb); s_h = 1.0 / math.sqrt(H)
    return {
        "E":  0.1 * mx.random.normal((k, emb), key=ks[0]),
        "Wx": s_in * mx.random.normal((H, emb), key=ks[1]),
        "Uh": s_h * mx.random.normal((H, H), key=ks[2]),
        "bh": mx.zeros((H,)),
        "h0": 0.1 * mx.random.normal((H,), key=ks[3]),
        "W":  s_h * mx.random.normal((n_classes, H), key=ks[4]), "b": mx.zeros((n_classes,)),
    }


def rnn_states(p: dict, x: mx.array) -> mx.array:
    """Elman tanh RNN hidden states [B,T,H]."""
    B, T = x.shape
    H = p["h0"].shape[0]
    e_all = p["E"][x]
    h = mx.broadcast_to(p["h0"], (B, H))
    hs = []
    for t in range(T):
        h = mx.tanh(e_all[:, t] @ p["Wx"].T + h @ p["Uh"].T + p["bh"])
        hs.append(h)
    return mx.stack(hs, axis=1)


def rnn_forward(p: dict, x: mx.array) -> mx.array:
    return rnn_states(p, x) @ p["W"].T + p["b"]


def lstm_init(k: int, H: int, n_classes: int, key, emb: int | None = None) -> dict:
    if emb is None:
        emb = H
    ks = mx.random.split(key, 11)
    s_in = 1.0 / math.sqrt(emb); s_h = 1.0 / math.sqrt(H)
    return {
        "E":  0.1 * mx.random.normal((k, emb), key=ks[0]),
        "Wf": s_in * mx.random.normal((H, emb), key=ks[1]), "Uf": s_h * mx.random.normal((H, H), key=ks[2]),
        "Wi": s_in * mx.random.normal((H, emb), key=ks[3]), "Ui": s_h * mx.random.normal((H, H), key=ks[4]),
        "Wg": s_in * mx.random.normal((H, emb), key=ks[5]), "Ug": s_h * mx.random.normal((H, H), key=ks[6]),
        "Wo": s_in * mx.random.normal((H, emb), key=ks[7]), "Uo": s_h * mx.random.normal((H, H), key=ks[8]),
        "bf": mx.ones((H,)),   # forget-gate bias 1.0 (standard LSTM init for stable long memory)
        "bi": mx.zeros((H,)), "bg": mx.zeros((H,)), "bo": mx.zeros((H,)),
        "h0": 0.1 * mx.random.normal((H,), key=ks[9]),
        "W":  s_h * mx.random.normal((n_classes, H), key=ks[10]), "b": mx.zeros((n_classes,)),
    }


def lstm_states(p: dict, x: mx.array) -> mx.array:
    """LSTM hidden (h) states [B,T,H]."""
    B, T = x.shape
    H = p["h0"].shape[0]
    e_all = p["E"][x]
    h = mx.broadcast_to(p["h0"], (B, H))
    c = mx.zeros((B, H))
    hs = []
    for t in range(T):
        e = e_all[:, t]
        f = mx.sigmoid(e @ p["Wf"].T + h @ p["Uf"].T + p["bf"])
        i = mx.sigmoid(e @ p["Wi"].T + h @ p["Ui"].T + p["bi"])
        g = mx.tanh(e @ p["Wg"].T + h @ p["Ug"].T + p["bg"])
        o = mx.sigmoid(e @ p["Wo"].T + h @ p["Uo"].T + p["bo"])
        c = f * c + i * g
        h = o * mx.tanh(c)
        hs.append(h)
    return mx.stack(hs, axis=1)


def lstm_forward(p: dict, x: mx.array) -> mx.array:
    return lstm_states(p, x) @ p["W"].T + p["b"]


# ------------- hidden-state extractors for the MECHANISM probe -------------
# Return the recurrent state [B,T,H] used by the readout, so we can measure whether the state
# collapses to a function of the true group element (the homomorphism the model must discover).

def gru_states(p: dict, x: mx.array) -> mx.array:
    B, T = x.shape
    H = p["h0"].shape[0]
    e_all = p["E"][x]
    h = mx.broadcast_to(p["h0"], (B, H))
    hs = []
    for t in range(T):
        e = e_all[:, t]
        z = mx.sigmoid(e @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
        r = mx.sigmoid(e @ p["Wr"].T + h @ p["Ur"].T + p["br"])
        n = mx.tanh(e @ p["Wn"].T + r * (h @ p["Un"].T) + p["bn"])
        h = (1.0 - z) * n + z * h
        hs.append(h)
    return mx.stack(hs, axis=1)


def pd_states(p: dict, x: mx.array) -> mx.array:
    B, T = x.shape
    H = p["d_raw"].shape[1]
    P = mx.softmax(p["P_logits"], axis=-1)
    d = mx.tanh(p["d_raw"])
    M = P * d[:, None, :]
    h = mx.broadcast_to(p["h0"], (B, H))
    hs = []
    for t in range(T):
        h = mx.einsum("bij,bj->bi", M[x[:, t]], h)
        hs.append(h)
    return mx.stack(hs, axis=1)


# ----------------------------- shared loss -----------------------------

def loss_fn(forward, p, x, y):
    """Mean cross-entropy over all positions. y: [B,T] int targets."""
    logits = forward(p, x)                           # [B,T,C]
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    yt = y.reshape(B * T)
    return mx.mean(nn.losses.cross_entropy(logits, yt))


def accuracy(forward, p, x, y):
    logits = forward(p, x)
    pred = mx.argmax(logits, axis=-1)
    return mx.mean((pred == y).astype(mx.float32))
