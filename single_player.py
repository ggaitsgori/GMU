from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from gmu_types import SinglePlayerSolutionMU


def solve_single_player_mu(
    a,
    b,
    R: float,
    tol: float = 1e-12,
) -> SinglePlayerSolutionMU:
    """
    Order-free solver for:
        max  sum_i a_i * x_i / (b_i + x_i)
        s.t. sum_i x_i = R,  x_i >= 0

    When b_i = 1 for all i this reduces to the standard single-player GMU problem.

    Parameters
    ----------
    a : array_like  -- positive project parameters
    b : array_like  -- positive denominators (same shape as a)
    R : float       -- total resource (>= 0)
    tol : float     -- numerical tolerance

    Returns
    -------
    SinglePlayerSolutionMU
        k_star : number of active projects
        c      : optimal marginal rate
        x      : optimal allocation vector
        v      : optimal objective value
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    n = a.size
    if n == 0:
        raise ValueError("Need at least one project.")
    if np.any(a <= 0):
        raise ValueError("All a_i must be positive.")
    if np.any(b <= 0):
        raise ValueError("All b_i must be positive.")
    if R < 0:
        raise ValueError("R must be nonnegative.")

    if R <= tol:
        return SinglePlayerSolutionMU(k_star=0, c=np.inf, x=np.zeros(n), v=0.0)

    y = np.sqrt(a * b)
    theta = np.sqrt(b / a)

    order = np.argsort(theta)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)

    a_s = a[order]
    b_s = b[order]
    y_s = y[order]
    th_s = theta[order]

    B = np.cumsum(b_s)
    Y = np.cumsum(y_s)

    k_star, t_star = None, None
    for k in range(1, n + 1):
        t_k = (R + B[k - 1]) / Y[k - 1]
        th_next = th_s[k] if k < n else np.inf
        if t_k <= th_next + tol:
            k_star, t_star = k, t_k
            break
    if k_star is None:
        k_star = n
        t_star = (R + B[-1]) / Y[-1]

    x_s = np.maximum(0.0, t_star * y_s - b_s)
    s = x_s.sum()
    if abs(s - R) > 10 * tol and s > tol:
        active = x_s > tol
        if np.any(active):
            x_s[active] *= R / s

    x = x_s[inv]
    k_active = int(np.sum(x > tol))
    c = 1.0 / (t_star * t_star)
    v = float(np.sum(a * x / (b + x)))

    return SinglePlayerSolutionMU(k_star=k_active, c=c, x=x, v=v)


# ---------------------------------------------------------------------------
# Math utilities used by other modules
# ---------------------------------------------------------------------------

def R_of_C(C: float, a, m: int, k: Optional[int] = None) -> float:
    """Compute the total resource R that corresponds to marginal rate C in a zone."""
    if C <= 0:
        raise ValueError("C must be strictly positive.")
    a = np.asarray(a, dtype=float)
    if k is None:
        k = len(a)
    if not (1 <= k <= len(a)):
        raise ValueError("k must be between 1 and len(a).")

    m1 = float(m) - 1.0
    S = sum(
        a[i] * (m1 / 2.0 + np.sqrt((m1 ** 2) / 4.0 + C / a[i]))
        for i in range(k)
    )
    return S / C - k


def C_of_R(
    R: float,
    a,
    m: int,
    k: Optional[int] = None,
    C_low_init: float = 1e-10,
    C_high_init: float = 100.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Invert R_of_C via bisection to find C given R."""
    R_target = float(R)
    if R_target < 0:
        raise ValueError("R must be nonnegative.")

    def f(C: float) -> float:
        return R_of_C(C, a, m, k) - R_target

    C_low = C_low_init
    while f(C_low) < 0:
        C_low *= 0.5

    C_high = C_high_init
    while f(C_high) > 0 and C_high < 1e12:
        C_high *= 2.0
    if f(C_high) > 0:
        raise RuntimeError("Failed to bracket root for C; try larger C_high_init.")

    for _ in range(max_iter):
        C_mid = 0.5 * (C_low + C_high)
        f_mid = f(C_mid)
        if abs(f_mid) < tol:
            return C_mid
        if f_mid > 0:
            C_low = C_mid
        else:
            C_high = C_mid

    return 0.5 * (C_low + C_high)


def project_load_and_rate(
    a: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    L_i = sum_j X[j,i]   (total load per project)
    C_i = a_i / (1 + L_i) (entry rate per project)
    """
    a = np.asarray(a, dtype=float)
    X = np.asarray(X, dtype=float)
    L = X.sum(axis=0)
    C = a / (1.0 + L)
    return L, C


def player_lambda_from_X(
    a: np.ndarray,
    X: np.ndarray,
    active_tol: float = 1e-10,
) -> np.ndarray:
    """
    Shadow rate λ_j for player j:
        d_{j,i} = a_i * (1 + L_i - x_{j,i}) / (1 + L_i)^2
    Returns the median of d_{j,i} over active projects (NaN if none active).
    """
    a = np.asarray(a, dtype=float)
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    L = X.sum(axis=0)
    denom = (1.0 + L) ** 2
    lam = np.full(m, np.nan, dtype=float)
    for j in range(m):
        active = np.where(X[j] > active_tol)[0]
        if active.size == 0:
            continue
        dji = a[active] * (1.0 + L[active] - X[j, active]) / denom[active]
        lam[j] = float(np.median(dji))
    return lam


def boundary_project_index_from_X(
    X: np.ndarray,
    active_tol: float = 1e-10,
) -> list:
    """
    For each player j, return the index of the first inactive project just beyond
    their support (0-based), or None if they are active on the last project.
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    out = []
    for j in range(m):
        idx = np.where(X[j] > active_tol)[0]
        if idx.size == 0:
            out.append(0 if n > 1 else None)
            continue
        k0 = int(idx.max())
        b = k0 + 1
        out.append(b if b < n else None)
    return out


def feasibility_report(
    X: np.ndarray,
    r_players: np.ndarray,
    *,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Quick sanity checks on an allocation matrix X:
      - nonnegativity
      - row sums match player resources
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    r = np.asarray(r_players, dtype=float).reshape(-1)
    row_sum = X.sum(axis=1)
    min_entry = float(X.min())
    neg_mass = float(np.abs(X[X < 0]).sum()) if np.any(X < 0) else 0.0
    row_err = float(np.max(np.abs(row_sum - r))) if row_sum.shape == r.shape else np.nan
    return {
        "min_entry": min_entry,
        "neg_mass": neg_mass,
        "max_row_sum_error": row_err,
        "row_sum_ok": float(row_err <= tol) if np.isfinite(row_err) else 0.0,
        "nonneg_ok": float(min_entry >= -tol),
    }


def compare_X(X1: np.ndarray, X2: np.ndarray) -> Dict[str, float]:
    """Element-wise comparison between two allocation matrices."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if X1.shape != X2.shape:
        return {"shape_match": 0.0}
    D = X1 - X2
    max_abs = float(np.max(np.abs(D)))
    denom = float(np.max(np.abs(X2))) or 1.0
    return {
        "shape_match": 1.0,
        "max_abs": max_abs,
        "max_rel_to_X2_max": float(max_abs / denom),
        "l1": float(np.sum(np.abs(D))),
        "l2": float(np.sqrt(np.sum(D * D))),
    }


def compute_player_rewards(
    a: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    u_j = sum_i a_i * x_{j,i} / (1 + L_i)
    Returns (rewards per player, total reward).
    """
    a = np.asarray(a, dtype=float)
    X = np.asarray(X, dtype=float)
    L = X.sum(axis=0)
    denom = 1.0 + L
    U = (X * (a / denom)[None, :]).sum(axis=1)
    return U, float(U.sum())
