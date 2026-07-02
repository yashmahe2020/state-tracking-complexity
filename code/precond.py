"""The per-transition Gram preconditioner (the new algorithm) + kappa(G) instrument.

THE OBJECT.  For a diagonal SSM, the transition parameters are {nu, omega}, each [k, d_c] (one row
per generator g). Writing the d_c-dim gradient rows side by side gives, per generator g, a vector

    v_g = concat( dL/dnu[g] , dL/domega[g] )  in  R^{2 d_c},     V = [v_1; ...; v_k]  in  R^{k x 2d_c}.

These k vectors all live in the SAME shared diagonal eigenspace, so they collide. Their Gram matrix

    G = V V^T   in   R^{k x k}

becomes ill-conditioned: the claim is  kappa(G) = Omega(k)  (Omega(k^2) for maximally non-abelian
groups) at matched parameter count. kappa(G) = lambda_max(G) / lambda_min(G).

THE FIX (training-time, inference-architecture-unchanged).  Whiten the gradient across generators:

    V_white = (G + eps I)^{-1/2} V      ->  V_white V_white^T = I  (decorrelated, O(1)-conditioned).

This is NOT natural gradient / K-FAC: G is built from the per-FSA-transition partition of the BPTT
gradient (keyed only by the observed input token g -- the same information Arm A receives; confound
Row 3), not from the Fisher of the output distribution over all parameters. The eigendecomposition is
on a k x k matrix (k <= 16) and is done in NumPy, OUTSIDE the autodiff path (it is a post-gradient
transform), so it is exact and microsecond-cheap.

Arm B      : apply V_white.
Arm B-norm : rescale V_white to ||V||_F (isolates the directional effect from the step-size effect;
             confound Row 1).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from models import TRANSITION_KEYS


def _stack_V(grads: dict) -> np.ndarray:
    """Stack transition-parameter gradient rows into V [k, 2*d_c] (NumPy)."""
    blocks = [np.asarray(grads[key], dtype=np.float64) for key in TRANSITION_KEYS]  # each [k, d_c]
    return np.concatenate(blocks, axis=1)  # [k, 2 d_c]


def gram_and_kappa(grads: dict) -> tuple[np.ndarray, float, float, float]:
    """Return (G, kappa, lambda_max, lambda_min) for the RAW (unregularised) Gram. Diagnostic."""
    V = _stack_V(grads)
    G = V @ V.T
    w = np.linalg.eigvalsh(G)  # ascending, real (G is symmetric PSD)
    lam_max = float(w[-1])
    lam_min = float(w[0])
    tiny = 1e-30
    kappa = lam_max / max(lam_min, tiny)
    return G, kappa, lam_max, lam_min


def precondition_transition_grads(grads: dict, eps: float, norm_match: bool = False) -> dict:
    """Return a NEW grads dict with {nu, omega} whitened by (G+eps*lmax I)^{-1/2}. Other keys untouched.

    eps is a SCALE-RELATIVE floor: the absolute regulariser is eps * lambda_max(G). This is
    scale-invariant (gradient magnitudes shrink over training, so an absolute eps would swamp G late
    in training and silently stop whitening). It caps the post-whitening condition number at ~1/eps
    while flooring numerically-null eigen-directions (confound Row 16).
    """
    widths = [grads[key].shape[1] for key in TRANSITION_KEYS]
    V = _stack_V(grads)                       # [k, sum(widths)]
    G = V @ V.T                                # [k, k]
    k = G.shape[0]
    w0 = np.linalg.eigvalsh(G)
    lam_max = max(float(w0[-1]), 1e-30)
    eps_abs = eps * lam_max
    w, U = np.linalg.eigh(G + eps_abs * np.eye(k))  # symmetric -> real eigh
    inv_sqrt = (U * (1.0 / np.sqrt(w))[None, :]) @ U.T   # (G+eps_abs I)^{-1/2}
    Vw = inv_sqrt @ V                         # [k, sum(widths)]
    if norm_match:
        fn_V = np.linalg.norm(V)
        fn_Vw = np.linalg.norm(Vw)
        if fn_Vw > 1e-30:
            Vw = Vw * (fn_V / fn_Vw)
    new = dict(grads)
    off = 0
    for key, wd in zip(TRANSITION_KEYS, widths):
        new[key] = mx.array(Vw[:, off:off + wd].astype(np.float32))
        off += wd
    return new
