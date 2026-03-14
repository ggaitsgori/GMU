from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gmu_types import (
    AlgoResult,
    GameSolutionBR,
    HistoryEntry,
    MarginalsResult,
    NECheckResult,
    OuterSnapshot,
    RestrictedGMUSpec,
    SinglePlayerSolutionMU,
    TwoPlayerFullSolution,
    TwoPlayerOneZoneSolution,
    TwoZoneSolutionForK,
    ZoneSolution,
    ZoneSpec,
)
from single_player import C_of_R, feasibility_report, compare_X, solve_single_player_mu


# ===========================================================================
# Best-response (Gauss-Seidel) algorithm
# ===========================================================================

def solve_gmu_best_response(
    A: np.ndarray,
    R_players: np.ndarray,
    X0: Optional[np.ndarray] = None,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    damping: float = 1.0,
    verbose: bool = False,
    track_history: bool = False,
    print_after_each_player: bool = False,
    print_precision: int = 4,
    print_max_rows: int = 50,
) -> GameSolutionBR:
    """
    Gauss-Seidel best-response iteration for the GMU game.

    Each player j sequentially solves their single-player problem with the
    others' current investments fixed.

    Parameters
    ----------
    A : (m, n) array  -- project parameters A[j, i] for player j, project i
    R_players : (m,) array -- total resource for each player
    X0 : optional starting allocation (m, n)
    damping : step size in (0, 1]; 1.0 = full step, smaller = conservative
    track_history : if True, store X after every player update in sol.history
    """
    A = np.asarray(A, dtype=float)
    R_players = np.asarray(R_players, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D (m_players, n_projects).")
    m, n = A.shape
    if R_players.shape != (m,):
        raise ValueError("R_players must have shape (m,).")
    if np.any(A <= 0):
        raise ValueError("All A[j,i] must be positive.")
    if np.any(R_players < 0):
        raise ValueError("All R_players must be nonnegative.")
    if not (0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1].")

    if X0 is None:
        X = np.zeros((m, n), dtype=float)
        for j in range(m):
            if R_players[j] > 0:
                X[j, :] = R_players[j] / n
    else:
        X = np.asarray(X0, dtype=float).copy()
        if X.shape != (m, n):
            raise ValueError("X0 must have shape (m, n).")
        for j in range(m):
            X[j, :] = np.maximum(0.0, X[j, :])
            s = X[j, :].sum()
            if s > 0:
                X[j, :] *= (R_players[j] / s) if R_players[j] > 0 else 0.0
            elif R_players[j] > 0:
                X[j, :] = R_players[j] / n

    T = X.sum(axis=0)
    hist_list = [X.copy()] if track_history else []
    converged = False
    it_used = 0

    def _maybe_print(step_it: int, player_j: int, max_update: float) -> None:
        if not print_after_each_player:
            return
        np.set_printoptions(precision=print_precision, suppress=True, linewidth=160)
        print(f"\n[iter={step_it}, player={player_j}] max_update_so_far={max_update:.3e}")
        rows = min(print_max_rows, m)
        print(X[:rows, :])
        if rows < m:
            print(f"... (showing first {rows} of {m} players)")

    for it in range(1, max_iter + 1):
        it_used = it
        max_update = 0.0

        for j in range(m):
            x_old = X[j, :].copy()
            b = 1.0 + (T - x_old)
            br = solve_single_player_mu(a=A[j, :], b=b, R=float(R_players[j]), tol=tol).x

            x_new = (1.0 - damping) * x_old + damping * br
            x_new = np.maximum(0.0, x_new)
            s = x_new.sum()
            if s > 0:
                x_new *= (R_players[j] / s) if R_players[j] > 0 else 0.0
            elif R_players[j] > 0:
                x_new[:] = R_players[j] / n

            X[j, :] = x_new
            T += x_new - x_old

            upd = float(np.max(np.abs(x_new - x_old)))
            if upd > max_update:
                max_update = upd

            if track_history:
                hist_list.append(X.copy())
            _maybe_print(it, j, max_update)

        if verbose and (it == 1 or it % 50 == 0):
            print(f"iter {it}: max_update={max_update:.3e}")

        if max_update <= tol:
            converged = True
            break

    utilities = np.sum(A * X / (1.0 + T), axis=1)
    history = np.stack(hist_list, axis=0) if track_history else None
    meta = {
        "shape": "(steps, m, n)",
        "steps_meaning": "step 0 is initial X; step k is after player (k-1) % m update",
        "m": m,
        "n": n,
    } if track_history else None

    return GameSolutionBR(
        X=X, utilities=utilities, iters=it_used, converged=converged,
        history=history, history_meta=meta,
    )


# ===========================================================================
# Two-player exact NE solver
# ===========================================================================

def solve_two_player_zone(
    a_zone,
    r1_zone: float,
    r2_zone: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> TwoPlayerOneZoneSolution:
    """Semi-restricted GMU solver for m=2 on a single fixed zone."""
    a = np.asarray(a_zone, dtype=float)
    k = len(a)
    if k == 0:
        raise ValueError("Zone must contain at least one project.")
    if np.any(a <= 0):
        raise ValueError("All a_i must be positive.")
    if r1_zone < 0 or r2_zone < 0:
        raise ValueError("Resources must be nonnegative.")
    if not np.all(a[:-1] >= a[1:] - tol):
        raise ValueError("a_zone must be nonincreasing.")

    R = r1_zone + r2_zone

    def sum_L(C: float):
        sqrt_term = np.sqrt(1.0 + 4.0 * C / a)
        L = (a / (2.0 * C)) * (1.0 + sqrt_term)
        return L, L.sum()

    def f(C: float) -> float:
        _, SL = sum_L(C)
        return SL - (R + k)

    C_low, C_high = 1e-12, 1.0
    while f(C_high) > 0 and C_high < 1e12:
        C_high *= 2.0
    if f(C_high) > 0:
        raise RuntimeError("Could not bracket root for C in two-player zone solver.")

    if f(C_low) * f(C_high) > 0:
        C_low, C_high = 1e-15, 1e15
        if f(C_low) * f(C_high) > 0:
            raise RuntimeError("Failed to bracket root for C.")

    C = 0.5 * (C_low + C_high)
    for _ in range(max_iter):
        C_mid = 0.5 * (C_low + C_high)
        f_mid = f(C_mid)
        if abs(f_mid) < tol:
            C = C_mid
            break
        if f_mid > 0:
            C_low = C_mid
        else:
            C_high = C_mid
    else:
        C = 0.5 * (C_low + C_high)

    L, _ = sum_L(C)
    denom = 2.0 * k + R
    c1 = C * (k + R - r1_zone) / denom
    c2 = C * (k + R - r2_zone) / denom
    x1 = L * (1 - c1 / C) - c1 / C
    x2 = L * (1 - c2 / C) - c2 / C
    return TwoPlayerOneZoneSolution(C=C, c1=c1, c2=c2, x1=x1, x2=x2, L=L)


def solve_two_zone_for_k(
    a,
    r1: float,
    r2: float,
    k: int,
    tol_root: float = 1e-10,
    max_iter: int = 200,
    tol_bc: float = 1e-8,
) -> Optional[TwoZoneSolutionForK]:
    """
    For a fixed k (common-zone cutoff), attempt to find a NE for the 2-player problem.
    Returns None if no NE exists at this k.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n.")
    if r1 < 0 or r2 < 0:
        raise ValueError("r1, r2 must be nonnegative.")

    if k == n:
        zone_sol = solve_two_player_zone(a, r1, r2)
        x1 = np.zeros(n, dtype=float)
        x2 = np.zeros(n, dtype=float)
        x1[:] = zone_sol.x1
        x2[:] = zone_sol.x2
        return TwoZoneSolutionForK(k2=n, k1=n, x1=x1, x2=x2, x_star=0.0,
                                   c1=zone_sol.c1, c2=zone_sol.c2)

    single_zone = solve_two_player_zone(a[:k], r1, r2)
    if a[k] <= single_zone.c1 + tol_bc and a[k] <= single_zone.c2 + tol_bc:
        x1 = np.zeros(n, dtype=float)
        x2 = np.zeros(n, dtype=float)
        x1[:k] = single_zone.x1
        x2[:k] = single_zone.x2
        return TwoZoneSolutionForK(k2=k, k1=k, x1=x1, x2=x2, x_star=0.0,
                                   c1=single_zone.c1, c2=single_zone.c2)

    a_zone2 = a[k:]

    def phi(x: float) -> float:
        sol1 = solve_two_player_zone(a[:k], r1 - x, r2)
        sol2 = solve_single_player_mu(a=a_zone2, b=np.ones(len(a_zone2)), R=x)
        return sol1.c1 - sol2.c

    x_min, x_max = 1e-10, r1 - r2
    if x_min >= x_max:
        return None

    phi_left, phi_right = phi(x_min), phi(x_max)
    if abs(phi_left) < tol_root:
        x_star = x_min
    elif abs(phi_right) < tol_root:
        x_star = x_max
    elif phi_left * phi_right > 0:
        return None
    else:
        xl, xr = x_min, x_max
        x_star = None
        for _ in range(max_iter):
            xm = 0.5 * (xl + xr)
            fm = phi(xm)
            if abs(fm) < tol_root:
                x_star = xm
                break
            if phi_left * fm > 0:
                xl = xm
            else:
                xr = xm
        if x_star is None:
            x_star = 0.5 * (xl + xr)

    sol1 = solve_two_player_zone(a[:k], r1 - x_star, r2)
    sol2 = solve_single_player_mu(a=a_zone2, b=np.ones(len(a_zone2)), R=x_star)

    x1 = np.zeros(n, dtype=float)
    x2 = np.zeros(n, dtype=float)
    x1[:k] = sol1.x1
    x2[:k] = sol1.x2
    x1[k:] = sol2.x

    positive_1 = np.where(x1 > tol_bc)[0]
    k1 = int(positive_1[-1]) + 1 if positive_1.size > 0 else 0

    for idx in range(k, n):
        if a[idx] / (1.0 + x1[idx]) > sol1.c2 + tol_bc:
            return None

    return TwoZoneSolutionForK(k2=k, k1=k1, x1=x1, x2=x2, x_star=x_star,
                                c1=sol1.c1, c2=sol1.c2)


def solve_two_player_full(
    a,
    r1: float,
    r2: float,
    tol_sort: float = 1e-12,
) -> TwoPlayerFullSolution:
    """
    Global 2-player NE solver for GMU. Tries k = 1, ..., n until a NE is found.

    Requires a nonincreasing (a_1 >= ... >= a_n > 0).
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    if n == 0:
        raise ValueError("Need at least one project.")
    if np.any(a <= 0):
        raise ValueError("All a_i must be positive.")
    if not np.all(a[:-1] >= a[1:] - tol_sort):
        raise ValueError("a must be nonincreasing: a_1 >= ... >= a_n.")
    if r1 < 0 or r2 < 0:
        raise ValueError("r1, r2 must be nonnegative.")

    for k in range(1, n + 1):
        sol_k = solve_two_zone_for_k(a, r1, r2, k)
        if sol_k is not None:
            return TwoPlayerFullSolution(
                k2=sol_k.k2, k1=sol_k.k1, x1=sol_k.x1, x2=sol_k.x2,
                c1=sol_k.c1, c2=sol_k.c2, x_star=sol_k.x_star,
            )

    raise RuntimeError("No NE found for any k in {1,...,n}.")


def check_ne_marginals(
    a,
    x1,
    x2,
    tol_active_equal: float = 1e-6,
    tol_inactive_leq: float = 1e-8,
    tol_active: float = 1e-10,
) -> NECheckResult:
    """Verify NE conditions via marginal utilities for the 2-player GMU game."""
    a = np.asarray(a, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n = len(a)
    if len(x1) != n or len(x2) != n:
        raise ValueError("a, x1, x2 must have the same length.")

    L = 1.0 + x1 + x2
    d1 = a * (L - x1) / (L * L)
    d2 = a * (L - x2) / (L * L)

    def check_player(d, x, name):
        active = x > tol_active
        info = {"active_indices": np.where(active)[0], "inactive_indices": np.where(~active)[0], "derivatives": d}
        if not np.any(active):
            info["reason"] = f"{name}: no active projects"
            return bool(np.all(d <= tol_inactive_leq)), None, info
        d_active = d[active]
        c_j = float(np.mean(d_active))
        max_dev = float(np.max(np.abs(d_active - c_j)))
        info["c"] = c_j
        info["max_dev_active"] = max_dev
        d_inactive = d[~active]
        max_inactive = float(np.max(d_inactive)) if d_inactive.size > 0 else -np.inf
        info["max_inactive_derivative"] = max_inactive
        ok = bool(max_dev <= tol_active_equal and np.all(d_inactive <= c_j + tol_inactive_leq))
        return ok, c_j, info

    ok1, c1, info1 = check_player(d1, x1, "player1")
    ok2, c2, info2 = check_player(d2, x2, "player2")
    return NECheckResult(is_ne=bool(ok1 and ok2), c1=c1, c2=c2, details={"player1": info1, "player2": info2})


# ===========================================================================
# m-player restricted GMU solver
# ===========================================================================

def solve_zone_m(
    a: np.ndarray,
    zone: ZoneSpec,
    r_zone: Dict[int, float],
    tol: float = 1e-10,
    max_iter: int = 200,
) -> ZoneSolution:
    """Solve a single zone with m_s players given per-player zone resources."""
    proj_idx = zone.project_indices
    players = zone.active_players
    m_s = len(players)
    k_s = len(proj_idx)
    if m_s == 0 or k_s == 0:
        raise ValueError("Zone must have at least one player and one project.")

    a_zone = a[proj_idx]
    R_s = sum(r_zone.get(j, 0.0) for j in players)
    if R_s < 0:
        raise ValueError("Negative column sum R_s.")

    C_s = C_of_R(R=R_s, a=a_zone, m=m_s, k=k_s, tol=tol, max_iter=max_iter)
    denom = m_s * k_s + (m_s - 1.0) * R_s

    c_per_player: Dict[int, float] = {}
    for j in players:
        r_js = r_zone.get(j, 0.0)
        beta_js = (k_s + R_s - r_js) / denom
        c_per_player[j] = beta_js * C_s

    return ZoneSolution(C=C_s, c_per_player=c_per_player)


class RestrictedGMUSolver:
    """
    Iterative MU solver for the restricted GMU game with fixed zones.
    Stages A (cross-swap + self-move) and B (column-level tightening).
    """

    def __init__(self, spec: RestrictedGMUSpec, enable_logging: bool = True):
        self.spec = spec
        self.a = spec.a
        self.r_total = spec.total_resources
        self.zones = spec.zones
        self.m = len(self.r_total)
        self.S = len(self.zones)
        self.r_zone = np.zeros((self.m, self.S), dtype=float)
        self.enable_logging = enable_logging
        self.history: List[HistoryEntry] = []

    def initialize_r_zone_uniform(self) -> None:
        """Split each player's resource equally across zones where they are active."""
        self.r_zone[:] = 0.0
        for j in range(self.m):
            active_zones = [s for s, zone in enumerate(self.zones) if j in zone.active_players]
            if not active_zones:
                continue
            share = self.r_total[j] / len(active_zones)
            for s in active_zones:
                self.r_zone[j, s] = share

    def compute_marginals(self) -> MarginalsResult:
        C_cols = np.zeros(self.S, dtype=float)
        c_rows = np.zeros((self.m, self.S), dtype=float)
        r_rows = np.zeros((self.m, self.S), dtype=float)

        for s, zone in enumerate(self.zones):
            r_zone_dict = {j: self.r_zone[j, s] for j in zone.active_players}
            sol = solve_zone_m(self.a, zone, r_zone_dict)
            C_cols[s] = sol.C
            for j, c_js in sol.c_per_player.items():
                c_rows[j, s] = c_js
                r_rows[j, s] = r_zone_dict[j]

        disruptions = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            active_s = [s for s, zone in enumerate(self.zones) if j in zone.active_players]
            if len(active_s) >= 2:
                vals = [c_rows[j, s] for s in active_s]
                disruptions[j] = max(vals) - min(vals)

        return MarginalsResult(
            C_cols=C_cols, c_rows=c_rows, r_rows=r_rows,
            disruptions=disruptions, global_disruption=float(np.max(disruptions)),
        )

    def _log_state(self, marg: MarginalsResult, iter_idx: int) -> None:
        if not self.enable_logging:
            return
        R_cols = np.sum(self.r_zone, axis=0)
        self.history.append(HistoryEntry(
            iter_idx=iter_idx,
            r_zone=self.r_zone.copy(),
            C_cols=marg.C_cols.copy(),
            c_rows=marg.c_rows.copy(),
            R_cols=R_cols.copy(),
        ))

    def _is_disruption_improved(
        self, marg_old: MarginalsResult, marg_new: MarginalsResult,
        tol_global: float = 1e-12, tol_sum: float = 1e-12,
    ) -> bool:
        g_old, g_new = float(marg_old.global_disruption), float(marg_new.global_disruption)
        if g_new < g_old - tol_global:
            return True
        if abs(g_new - g_old) <= tol_global:
            if float(np.sum(marg_new.disruptions)) < float(np.sum(marg_old.disruptions)) - tol_sum:
                return True
        return False

    def _attempt_cross_swap(
        self, marg: MarginalsResult,
        tol_global: float = 1e-12, tol_sum: float = 1e-12, max_backtracking: int = 10,
    ) -> bool:
        c_rows = marg.c_rows
        for j in range(self.m):
            active_s_j = [s for s, zone in enumerate(self.zones) if j in zone.active_players]
            if len(active_s_j) < 2:
                continue
            for k in active_s_j:
                for ℓ in active_s_j:
                    if k == ℓ:
                        continue
                    diff_j = c_rows[j, k] - c_rows[j, ℓ]
                    if diff_j <= 0:
                        continue
                    for i in range(self.m):
                        if i == j:
                            continue
                        if i not in self.zones[k].active_players or i not in self.zones[ℓ].active_players:
                            continue
                        if c_rows[i, k] - c_rows[i, ℓ] >= 0:
                            continue
                        max_tau = min(self.r_zone[j, ℓ], self.r_zone[i, k])
                        if max_tau <= 0:
                            continue
                        tau = max_tau
                        for _ in range(max_backtracking):
                            r_old = self.r_zone.copy()
                            self.r_zone[j, k] += tau
                            self.r_zone[j, ℓ] -= tau
                            self.r_zone[i, k] -= tau
                            self.r_zone[i, ℓ] += tau
                            marg_new = self.compute_marginals()
                            if self._is_disruption_improved(marg, marg_new, tol_global, tol_sum):
                                return True
                            self.r_zone = r_old
                            tau *= 0.5
                            if tau < 1e-12:
                                break
            return False
        return False

    def _attempt_self_move(
        self, marg: MarginalsResult,
        tol_global: float = 1e-12, tol_sum: float = 1e-12, max_backtracking: int = 10,
    ) -> bool:
        disruptions, c_rows = marg.disruptions, marg.c_rows
        j_star = int(np.argmax(disruptions))
        if disruptions[j_star] <= 0:
            return False
        active_s = [s for s, zone in enumerate(self.zones) if j_star in zone.active_players]
        if len(active_s) < 2:
            return False
        positive_s = [s for s in active_s if self.r_zone[j_star, s] > 0]
        if not positive_s:
            return False
        s_right = max(positive_s)
        c_at_s_right = c_rows[j_star, s_right]
        s_min = min(active_s, key=lambda s: c_rows[j_star, s])
        s_max = max(active_s, key=lambda s: c_rows[j_star, s])
        if c_at_s_right > c_rows[j_star, s_min]:
            s_target = s_min
        elif c_at_s_right < c_rows[j_star, s_max]:
            s_target = s_max
        else:
            return False
        tau = self.r_zone[j_star, s_right]
        if tau <= 0:
            return False
        for _ in range(max_backtracking):
            r_old = self.r_zone.copy()
            self.r_zone[j_star, s_right] -= tau
            self.r_zone[j_star, s_target] += tau
            marg_new = self.compute_marginals()
            if self._is_disruption_improved(marg, marg_new, tol_global, tol_sum):
                return True
            self.r_zone = r_old
            tau *= 0.5
            if tau < 1e-12:
                break
        return False

    def step_A(self, tol_disruption: float = 1e-6, max_inner_iter: int = 1000) -> bool:
        """Run cross-swap and self-move until disruption is within tolerance."""
        changed = False
        for _ in range(max_inner_iter):
            marg = self.compute_marginals()
            if marg.global_disruption <= tol_disruption:
                return changed
            if self._attempt_cross_swap(marg):
                changed = True
                continue
            if self._attempt_self_move(marg):
                changed = True
                continue
            break
        return changed

    def step_B(
        self,
        tol_row_improve: float = 1e-9,
        tol_global_increase: float = 1e-9,
        max_backtracking: int = 15,
    ) -> bool:
        """Column-level resource tightening (Stage B)."""
        marg_old = self.compute_marginals()
        c_rows_old = marg_old.c_rows
        d_old = float(marg_old.global_disruption)
        if d_old <= 0.0:
            return False

        def row_spread(j: int, marg: MarginalsResult) -> float:
            vals = [marg.c_rows[j, s] for s, zone in enumerate(self.zones) if j in zone.active_players]
            return max(vals) - min(vals) if len(vals) >= 2 else 0.0

        for s in range(self.S):
            zone_s = self.zones[s]
            J, L_map = [], {}
            for j in zone_s.active_players:
                c_js = c_rows_old[j, s]
                best_l = min(
                    (l for l in range(self.S)
                     if l != s and j in self.zones[l].active_players
                     and self.r_zone[j, l] > 0 and c_rows_old[j, l] < c_js - 1e-12),
                    key=lambda l: c_rows_old[j, l],
                    default=None,
                )
                if best_l is not None:
                    J.append(j)
                    L_map[j] = best_l
            if not J:
                continue
            max_eps = min(self.r_zone[j, L_map[j]] for j in J)
            if max_eps <= 0.0:
                continue
            spreads_old = {j: row_spread(j, marg_old) for j in J}
            eps = max_eps
            for _ in range(max_backtracking):
                r_old = self.r_zone.copy()
                for j in J:
                    self.r_zone[j, s] += eps
                    self.r_zone[j, L_map[j]] -= eps
                marg_new = self.compute_marginals()
                all_improved = all(row_spread(j, marg_new) < spreads_old[j] - tol_row_improve for j in J)
                if all_improved and float(marg_new.global_disruption) <= d_old + tol_global_increase:
                    return True
                self.r_zone = r_old
                eps *= 0.5
                if eps < 1e-12:
                    break
        return False

    def run(self, tol_disruption: float = 1e-6, max_outer_iter: int = 100) -> MarginalsResult:
        for outer in range(max_outer_iter):
            marg = self.compute_marginals()
            self._log_state(marg, iter_idx=outer)
            if marg.global_disruption <= tol_disruption:
                return marg
            if self.step_A(tol_disruption=tol_disruption):
                continue
            if not self.step_B():
                return marg
        marg = self.compute_marginals()
        self._log_state(marg, iter_idx=max_outer_iter)
        return marg


# ===========================================================================
# Zone construction helpers
# ===========================================================================

def enumerate_k_configurations(m: int, n: int) -> List[List[int]]:
    """
    Enumerate all admissible cutoff vectors k = (k_1,...,k_m) with
    1 <= k_m <= ... <= k_1 <= n.
    """
    configs: List[List[int]] = []
    cur = [1] * m

    def backtrack(pos: int, last_val: int) -> None:
        if pos == m:
            configs.append(cur[::-1])
            return
        for v in range(last_val, n + 1):
            cur[pos] = v
            backtrack(pos + 1, v)

    backtrack(0, 1)
    return configs


def build_zones_from_k(k: List[int], m: int, n: int) -> List[ZoneSpec]:
    """Build zone specs from a cutoff vector k (1-based)."""
    zones: List[ZoneSpec] = []
    current_players: Optional[List[int]] = None
    start: Optional[int] = None

    for i in range(n):
        active = [j for j in range(m) if k[j] >= (i + 1)]
        if active:
            if current_players is None:
                current_players, start = active, i
            elif active != current_players:
                zones.append(ZoneSpec(project_indices=list(range(start, i)), active_players=current_players))
                current_players, start = active, i
        else:
            if current_players is not None:
                zones.append(ZoneSpec(project_indices=list(range(start, i)), active_players=current_players))
                current_players = start = None

    if current_players is not None and start is not None:
        zones.append(ZoneSpec(project_indices=list(range(start, n)), active_players=current_players))

    return zones


def allocate_within_zone(
    a: np.ndarray,
    zone: ZoneSpec,
    r_zone_s: Dict[int, float],
    C_s: float,
    c_per_player_s: Optional[Dict[int, float]] = None,
    clamp_negative: bool = True,
    tol: float = 1e-10,
) -> Dict[int, np.ndarray]:
    """Compute per-project allocations x_{j,i} inside a zone from its aggregate C_s."""
    proj_idx = list(zone.project_indices)
    k = len(proj_idx)
    players = list(zone.active_players)
    m_zone = len(players)

    if k == 0 or m_zone == 0:
        return {j: np.zeros(0, dtype=float) for j in players}

    a_zone = np.asarray(a[proj_idx], dtype=float)
    R_s = sum(float(r_zone_s.get(j, 0.0)) for j in players)

    if R_s <= 0.0 or C_s <= 0.0:
        return {j: np.zeros(k, dtype=float) for j in players}

    m1 = float(m_zone - 1)
    L = np.array([
        a_zone[idx] / C_s * (m1 / 2.0 + np.sqrt(max(0.0, (m1 ** 2) / 4.0 + C_s / a_zone[idx])))
        for idx in range(k)
    ])

    S1 = float(np.sum(L))
    S2 = float(np.sum(L * L / a_zone))

    if S2 <= 0.0:
        return {j: np.zeros(k, dtype=float) for j in players}

    x_zone: Dict[int, np.ndarray] = {}
    for j in players:
        rj = float(r_zone_s.get(j, 0.0))
        c_j = (S1 - rj) / S2
        x_j = L - c_j * (L * L / a_zone)
        if clamp_negative:
            x_j[x_j < 0.0] = 0.0
        x_zone[j] = x_j

    return x_zone


def compute_full_allocation_from_solver(
    solver: RestrictedGMUSolver,
    marg: MarginalsResult,
    a: np.ndarray,
) -> np.ndarray:
    """Compute per-project allocations X[j, i] from a solved RestrictedGMUSolver."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    m = solver.m
    X = np.zeros((m, n), dtype=float)

    for s, zone in enumerate(solver.zones):
        r_zone_s = {j: solver.r_zone[j, s] for j in zone.active_players}
        x_zone = allocate_within_zone(a, zone, r_zone_s, marg.C_cols[s])
        for j in zone.active_players:
            X[j, np.array(zone.project_indices)] = x_zone[j]

    return X


def compute_true_bc_penalty(
    solver: RestrictedGMUSolver,
    marg: MarginalsResult,
    a: np.ndarray,
    X: np.ndarray,
    active_tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:
    """
    Compute boundary-condition violation:
      violation_j = max(0, a[k_j+1] / (1 + L_{-j, k_j+1}) - c_j_last)
    where k_j is the last active project of player j (0-based) and L_{-j,i}
    is the total load from other players on project i.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    c_rows = marg.c_rows
    m = solver.m

    proj_to_zone = [-1] * n
    for s, zone in enumerate(solver.zones):
        for i in zone.project_indices:
            proj_to_zone[i] = s

    bc_by_player = np.zeros(m, dtype=float)
    for j in range(m):
        active_proj = [i for i in range(n) if X[j, i] > active_tol]
        if not active_proj:
            continue
        k_j = max(active_proj)
        boundary_i = k_j + 1
        if boundary_i >= n:
            continue
        s_last = proj_to_zone[k_j]
        if s_last < 0:
            continue
        c_j_last = c_rows[j, s_last]
        L_minus_j = float(np.sum(X[:, boundary_i]) - X[j, boundary_i])
        bc_by_player[j] = max(0.0, a[boundary_i] / (1.0 + L_minus_j) - c_j_last)

    return float(np.max(bc_by_player)), bc_by_player


def run_mug_restricted_for_k(
    a,
    r_players,
    k: List[int],
    max_iter: int = 500,
    tol_row: float = 1e-6,
    enable_logging: bool = False,
) -> Tuple[RestrictedGMUSolver, MarginalsResult, np.ndarray, float, float, np.ndarray]:
    """Build zones from k, run RestrictedGMUSolver, return (solver, marg, X, d_rows, bc_penalty, bc_by_player)."""
    a = np.asarray(a, dtype=float)
    r_players = np.asarray(r_players, dtype=float)
    n = len(a)
    m = len(r_players)
    if len(k) != m:
        raise ValueError("len(k) must equal number of players.")

    zones = build_zones_from_k(list(k), m=m, n=n)
    if not zones:
        raise ValueError("No zones generated from k.")

    spec = RestrictedGMUSpec(a=a, total_resources=r_players, zones=zones)
    solver = RestrictedGMUSolver(spec, enable_logging=enable_logging)
    solver.initialize_r_zone_uniform()
    marg = solver.run(tol_disruption=tol_row, max_outer_iter=max_iter)
    X = compute_full_allocation_from_solver(solver, marg, a)
    bc_penalty, bc_by_player = compute_true_bc_penalty(solver, marg, a, X)
    return solver, marg, X, float(marg.global_disruption), bc_penalty, bc_by_player


# ===========================================================================
# Global NE search: greedy k-search
# ===========================================================================

def find_global_NE_by_k_search(
    a,
    r_players,
    max_iter_inner: int = 500,
    tol_row_inner: float = 1e-6,
    bc_tol_equiv: float = 1e-8,
    verbose: bool = True,
    max_k_steps: Optional[int] = None,
    record_mode: Optional[str] = None,
):
    """
    Greedy global NE search over cutoff vectors k, guided by BC violations.

    record_mode: None | "between" (last inner snapshot per k-step) | "full" (all inner snapshots)
    Returns: (best_solver, best_marg, best_k, best_bc_penalty, best_disruption, X_best, global_frames)
    """
    from gmu_types import GlobalFrame

    a = np.asarray(a, dtype=float)
    r_players = np.asarray(r_players, dtype=float)
    n = len(a)
    m = len(r_players)
    if max_k_steps is None:
        max_k_steps = m * n

    k = [1] * m
    visited = {tuple(k)}
    best_solver = best_marg = X_best = None
    best_k = k.copy()
    best_bc_penalty = best_disruption = np.inf
    global_frames: list = []

    for step in range(max_k_steps):
        if verbose:
            print(f"\n=== k-step {step+1}/{max_k_steps} | k={k} ===")

        solver, marg, X, d_rows, bc_penalty, bc_by_player = run_mug_restricted_for_k(
            a, r_players, k,
            max_iter=max_iter_inner, tol_row=tol_row_inner,
            enable_logging=(record_mode is not None),
        )

        if record_mode is not None:
            zones = build_zones_from_k(k, m=m, n=n)
            if record_mode == "between" and solver.history:
                global_frames.append(GlobalFrame(step_idx=step, k=k.copy(), zones=zones, entry=solver.history[-1]))
            elif record_mode == "full":
                for entry in solver.history:
                    global_frames.append(GlobalFrame(step_idx=step, k=k.copy(), zones=zones, entry=entry))

        if verbose:
            print(f"  BC={bc_penalty:.6e}  d_rows={d_rows:.6e}")

        if bc_penalty + bc_tol_equiv < best_bc_penalty or (
            abs(bc_penalty - best_bc_penalty) <= bc_tol_equiv and d_rows < best_disruption
        ):
            best_bc_penalty = bc_penalty
            best_disruption = d_rows
            best_solver, best_marg = solver, marg
            best_k = k.copy()
            X_best = X.copy()

        if bc_penalty <= bc_tol_equiv and d_rows <= tol_row_inner:
            if verbose:
                print("Converged.")
            break

        bc_by_player = np.asarray(bc_by_player, dtype=float)
        candidates = [
            j for j in range(m)
            if bc_by_player[j] > 0.0
            and k[j] < n
            and (j == 0 or k[j] + 1 <= k[j - 1])
            and tuple(k[:j] + [k[j] + 1] + k[j+1:]) not in visited
        ]
        if not candidates:
            if verbose:
                print("No expansion candidates; stopping.")
            break

        j_star = max(candidates, key=lambda j: bc_by_player[j])
        if verbose:
            print(f"  Expanding j={j_star}: k[j]={k[j_star]} -> {k[j_star]+1}")
        k[j_star] += 1
        visited.add(tuple(k))

    if best_solver is None:
        raise RuntimeError("Greedy NE search failed to find any feasible configuration.")
    if verbose:
        print(f"\n=== Done | best_k={best_k} | BC={best_bc_penalty:.3e} | d_rows={best_disruption:.3e} ===")

    return best_solver, best_marg, best_k, best_bc_penalty, best_disruption, X_best, global_frames


# ===========================================================================
# Global NE search: monotone-projects algorithm
# ===========================================================================

def _active_in_project(X: np.ndarray, j: int, proj_i: int, eps_mass: float) -> bool:
    if proj_i < 0 or proj_i >= X.shape[1]:
        return False
    return float(X[j, proj_i]) > eps_mass


def _eligible_by_prev_project_activity(X: np.ndarray, j: int, target_K: int, eps_mass: float) -> bool:
    """Monotonicity gate: player j can enter project target_K (1-based) only if active in project target_K-1."""
    if target_K <= 1:
        return True
    return _active_in_project(X, j, target_K - 2, eps_mass)  # target_K-1 in 1-based = target_K-2 in 0-based


def k_from_X(X: np.ndarray, eps: float = 1e-10) -> list:
    """Last active project index + 1 (1-based) for each player. Returns at least 1."""
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    out = []
    for j in range(m):
        pos = np.where(X[j] > eps)[0]
        out.append(int(pos.max() + 1) if pos.size else 1)
    return out


def _proj_to_zone_map(solver: RestrictedGMUSolver) -> list:
    n = max(max(z.project_indices) for z in solver.zones) + 1 if solver.zones else 0
    proj_to_zone = [-1] * n
    for s, zone in enumerate(solver.zones):
        for i in zone.project_indices:
            if 0 <= i < n:
                proj_to_zone[i] = s
    return proj_to_zone


def bc_details_df(
    solver: RestrictedGMUSolver,
    marg: MarginalsResult,
    a: np.ndarray,
    X: np.ndarray,
    r_players: np.ndarray,
    *,
    active_tol: float = 1e-10,
    k_perm: Optional[list] = None,
    stopped: Optional[list] = None,
    cap: Optional[list] = None,
) -> "pd.DataFrame":
    """Build a DataFrame with per-player BC diagnostic info."""
    a = np.asarray(a, dtype=float)
    X = np.asarray(X, dtype=float)
    r_players = np.asarray(r_players, dtype=float).reshape(-1)
    m, n = X.shape
    c_rows = marg.c_rows
    proj_to_zone = _proj_to_zone_map(solver)
    if len(proj_to_zone) < n:
        proj_to_zone = proj_to_zone + [-1] * (n - len(proj_to_zone))

    k_act = k_from_X(X, eps=active_tol)
    rows = []
    for j in range(m):
        active_proj = np.where(X[j] > active_tol)[0]
        if active_proj.size == 0 or int(active_proj.max()) + 1 >= n:
            viol = 0.0
            row = dict(j=j, r=float(r_players[j]),
                       k_perm=None if k_perm is None else int(k_perm[j]),
                       k_active=int(k_act[j]),
                       stopped=None if stopped is None else bool(stopped[j]),
                       cap=None if cap is None else int(cap[j]),
                       last_active_1b=None if active_proj.size == 0 else int(active_proj.max() + 1),
                       boundary_1b=None, s_last=None, c_j_last=None,
                       L_minus_j=None, deriv_boundary=None, violation=viol)
        else:
            k_j = int(active_proj.max())
            boundary_i = k_j + 1
            s_last = proj_to_zone[k_j]
            c_j_last = float(c_rows[j, s_last]) if s_last >= 0 else np.nan
            L_minus_j = float(np.sum(X[:, boundary_i]) - X[j, boundary_i])
            deriv_boundary = float(a[boundary_i] / (1.0 + L_minus_j))
            viol = float(max(0.0, deriv_boundary - c_j_last))
            row = dict(j=j, r=float(r_players[j]),
                       k_perm=None if k_perm is None else int(k_perm[j]),
                       k_active=int(k_act[j]),
                       stopped=None if stopped is None else bool(stopped[j]),
                       cap=None if cap is None else int(cap[j]),
                       last_active_1b=int(k_j + 1), boundary_1b=int(boundary_i + 1),
                       s_last=s_last, c_j_last=c_j_last,
                       L_minus_j=L_minus_j, deriv_boundary=deriv_boundary, violation=viol)
        rows.append(row)
    return pd.DataFrame(rows)


def print_step_debug(
    *, title: str, step: int, k_perm: list, stopped: list, cap: list,
    bc_penalty: float, bc_by_player: np.ndarray, d_rows: float,
    solver, marg, a: np.ndarray, X: np.ndarray, r_players: np.ndarray,
    active_tol: float = 1e-10, show_df: bool = True,
) -> None:
    print(f"\n=== {title} | step={step} ===")
    print(f"k_perm={k_perm} | k_active={k_from_X(X, eps=active_tol)}")
    print(f"stopped={stopped} | cap={cap}")
    print(f"bc_penalty={bc_penalty:.6e} | bc_by_player={np.asarray(bc_by_player, float)} | d_rows={d_rows:.6e}")
    if show_df:
        df = bc_details_df(solver, marg, a, X, r_players, active_tol=active_tol,
                           k_perm=k_perm, stopped=stopped, cap=cap)
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
            print(df)


def _solve_restricted(
    a: np.ndarray,
    r_players: np.ndarray,
    k: List[int],
    *,
    max_iter_inner: int,
    tol_row_inner: float,
    enable_logging: bool,
) -> Tuple[Any, Any, np.ndarray, float, float, np.ndarray]:
    solver, marg, X, d_rows, bc_penalty, bc_by_player = run_mug_restricted_for_k(
        a, r_players, k,
        max_iter=max_iter_inner, tol_row=tol_row_inner, enable_logging=enable_logging,
    )
    return solver, marg, X, float(d_rows), float(bc_penalty), np.asarray(bc_by_player, dtype=float)


def find_global_NE_monotone_projects(
    a: np.ndarray,
    r_players: np.ndarray,
    *,
    max_iter_inner: int = 500,
    tol_row_inner: float = 1e-6,
    bc_tol: float = 1e-8,
    bc_tol_player: Optional[float] = None,
    eps_mass: float = 1e-10,
    max_outer_steps: Optional[int] = None,
    enable_logging: bool = False,
    verbose: bool = True,
    store_X_in_history: bool = True,
    debug: bool = True,
    debug_show_df: bool = True,
    debug_candidates: bool = True,
) -> Tuple[Any, Any, list, float, float, np.ndarray, List[OuterSnapshot]]:
    """
    Main algorithm: monotone-projects NE search.

    Players are ordered by decreasing resource. The algorithm opens projects
    one by one, expands existing players' supports if their BC is violated,
    and applies monotonicity / refusal caps to lower-ranked players.

    Returns
    -------
    solver, marg, k, bc_penalty, d_rows, X, history
    """
    frame_id = 0
    a = np.asarray(a, dtype=float)
    r_players = np.asarray(r_players, dtype=float).reshape(-1)
    m = int(r_players.shape[0])
    n = int(a.shape[0])

    if bc_tol_player is None:
        bc_tol_player = bc_tol

    order = list(np.argsort(-r_players))
    rank = {j: i for i, j in enumerate(order)}

    k: list = [1] * m
    stopped: list = [False] * m
    cap: list = [n] * m
    history: List[OuterSnapshot] = []

    def bc_wants_expand(bc_by_player: np.ndarray, j: int) -> bool:
        return float(bc_by_player[j]) > float(bc_tol_player)

    def _apply_refusal_cap(j_refuse: int, refused_K: int) -> None:
        cap_value = refused_K - 1
        r_ref = rank[j_refuse]
        for jj in order:
            if rank[jj] > r_ref:
                cap[jj] = min(cap[jj], cap_value)
                if k[jj] > cap[jj]:
                    k[jj] = cap[jj]

    def _record(step: int, phase: str, player: Optional[int], target_K: Optional[int],
                solver: Any, X: np.ndarray, d_rows: float, bc_penalty: float,
                bc_by_player: np.ndarray, note: str = "") -> None:
        entry = solver.history[-1] if (hasattr(solver, "history") and solver.history) else None
        history.append(OuterSnapshot(
            step=step, phase=phase, player=player, target_K=target_K,
            k=list(k), stopped=list(stopped),
            bc_penalty=float(bc_penalty),
            bc_by_player=np.asarray(bc_by_player, dtype=float).copy(),
            d_rows=float(d_rows),
            note=note + f" | cap={cap}",
            X=(X.copy() if store_X_in_history else None),
            solver_last_entry=entry,
            frame_id=frame_id,
        ))

    # Initial solve
    solver, marg, X, d_rows, bc_penalty, bc_by_player = _solve_restricted(
        a, r_players, k, max_iter_inner=max_iter_inner, tol_row_inner=tol_row_inner,
        enable_logging=enable_logging,
    )
    _record(0, "solve", None, None, solver, X, d_rows, bc_penalty, bc_by_player)
    if verbose:
        print(f"[init] k={k}  bc={bc_penalty:.3e}  d_rows={d_rows:.3e}")
    if debug:
        print_step_debug(title="AFTER SOLVE (init)", step=0, k_perm=k, stopped=stopped, cap=cap,
                         bc_penalty=bc_penalty, bc_by_player=bc_by_player, d_rows=d_rows,
                         solver=solver, marg=marg, a=a, X=X, r_players=r_players,
                         active_tol=eps_mass, show_df=debug_show_df)

    step = 0
    while bc_penalty > bc_tol:
        step += 1
        if max_outer_steps is not None and step > max_outer_steps:
            raise RuntimeError(f"Exceeded max_outer_steps={max_outer_steps}.")

        K = max(k)

        # Phase A: expand existing players to K
        did_expand_existing = False
        for j in order:
            if stopped[j] or cap[j] < K or k[j] >= K:
                continue

            if not _eligible_by_prev_project_activity(X, j, K, eps_mass):
                stopped[j] = True
                _apply_refusal_cap(j, K)
                _record(step, "stop_ineligible_existing", j, K, solver, X, d_rows, bc_penalty, bc_by_player,
                        note="not active in previous project")
                if verbose:
                    print(f"[step {step}] stop_ineligible_existing j={j} K={K}")
                break

            if debug and debug_candidates:
                print(f"\n[step {step}] consider expand_existing j={j}: k[j]={k[j]} -> {K}, "
                      f"bc[j]={bc_by_player[j]:.3e}")

            if not bc_wants_expand(bc_by_player, j):
                stopped[j] = True
                _apply_refusal_cap(j, K)
                _record(step, "stop_existing_bc_satisfied", j, K, solver, X, d_rows, bc_penalty, bc_by_player)
                if verbose:
                    print(f"[step {step}] stop_existing_bc_satisfied j={j} K={K}")
                break

            k_trial = list(k)
            k_trial[j] = min(k[j] + 1, K, cap[j])
            solver_t, marg_t, X_t, d_rows_t, bc_t, bc_by_t = _solve_restricted(
                a, r_players, k_trial, max_iter_inner=max_iter_inner,
                tol_row_inner=tol_row_inner, enable_logging=enable_logging,
            )
            improved = (bc_t < bc_penalty - 1e-12) or (bc_by_t[j] < bc_by_player[j] - 1e-12)

            if improved:
                k = k_trial
                solver, marg, X, d_rows, bc_penalty, bc_by_player = solver_t, marg_t, X_t, d_rows_t, bc_t, bc_by_t
                _record(step, "expand_existing", j, k_trial[j], solver, X, d_rows, bc_penalty, bc_by_player)
                did_expand_existing = True
                if verbose:
                    print(f"[step {step}] expand_existing j={j} -> k[j]={k_trial[j]}  bc={bc_penalty:.3e}")
                break

        if did_expand_existing:
            continue

        # Phase B: open new project
        if K >= n:
            if verbose:
                print(f"[step {step}] K=n={n}, fallback to full access.")
            k_full = [n] * m
            solver, marg, X, d_rows, bc_penalty, bc_by_player = _solve_restricted(
                a, r_players, k_full, max_iter_inner=max_iter_inner,
                tol_row_inner=tol_row_inner, enable_logging=enable_logging,
            )
            k = list(k_full)
            _record(step, "fallback_full_access", None, n, solver, X, d_rows, bc_penalty, bc_by_player)
            break

        target_K = K + 1
        candidates = [
            j for j in order
            if not stopped[j] and cap[j] >= target_K
            and _active_in_project(X, j, K - 1, eps_mass)
        ]

        if not candidates:
            raise RuntimeError(f"No candidates to open new project {target_K} from frontier {K}.")

        seeded_any = False
        for j in candidates:
            if bc_wants_expand(bc_by_player, j):
                k_trial = list(k)
                k_trial[j] = min(target_K, cap[j])
                solver_t, marg_t, X_t, d_rows_t, bc_t, bc_by_t = _solve_restricted(
                    a, r_players, k_trial, max_iter_inner=max_iter_inner,
                    tol_row_inner=tol_row_inner, enable_logging=enable_logging,
                )
                k = k_trial
                solver, marg, X, d_rows, bc_penalty, bc_by_player = solver_t, marg_t, X_t, d_rows_t, bc_t, bc_by_t
                _record(step, "expand_new", j, target_K, solver, X, d_rows, bc_penalty, bc_by_player)
                seeded_any = True
                if verbose:
                    print(f"[step {step}] expand_new j={j} -> k[j]={target_K}  bc={bc_penalty:.3e}")
                continue

            # Refusal
            stopped[j] = True
            _apply_refusal_cap(j, target_K)
            _record(step, "refuse_new_bc_satisfied", j, target_K, solver, X, d_rows, bc_penalty, bc_by_player)
            if verbose:
                print(f"[step {step}] refuse_new j={j} at K={target_K}")
            break

        if not seeded_any:
            raise RuntimeError(f"No one entered new project {target_K} but bc_penalty={bc_penalty:.3e}.")

    if verbose:
        print(f"[done] step={step}  k={k}  bc={bc_penalty:.3e}  d_rows={d_rows:.3e}")

    return solver, marg, k, bc_penalty, d_rows, X, history


# ===========================================================================
# Algorithm comparison utilities
# ===========================================================================

def normalize_algo_output(
    name: str,
    out: Any,
    *,
    expect_X_key: str = "X",
) -> AlgoResult:
    """Normalize output from any algorithm into a standard AlgoResult."""
    if isinstance(out, dict):
        X = np.asarray(out.get(expect_X_key), dtype=float)
        k_raw = out.get("k", None)
        k = np.asarray(k_raw, dtype=int).reshape(-1) if k_raw is not None else None
        return AlgoResult(
            name=name, k=k, X=X,
            bc_penalty=float(out["bc_penalty"]) if "bc_penalty" in out else None,
            d_rows=float(out["d_rows"]) if "d_rows" in out else None,
            solver=out.get("solver"), marg=out.get("marg"), history=out.get("history"), raw=out,
        )

    if isinstance(out, (tuple, list)):
        L = len(out)
        if L == 7:
            solver, marg, k, bc_penalty, d_rows, X, history = out
            return AlgoResult(
                name=name, k=np.asarray(k, dtype=int).reshape(-1) if k is not None else None,
                X=np.asarray(X, dtype=float),
                bc_penalty=float(bc_penalty), d_rows=float(d_rows),
                solver=solver, marg=marg, history=history, raw=out,
            )
        for i in [-1, -2]:
            try:
                X = np.asarray(out[i], dtype=float)
                if X.ndim == 2:
                    return AlgoResult(name=name, k=None, X=X, bc_penalty=None, d_rows=None, raw=out)
            except Exception:
                pass
        raise ValueError(f"Cannot parse output tuple of length {L} for {name}.")

    raise ValueError(f"Unsupported output type for {name}: {type(out)}")


def compare_algorithms(
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    new_algo: Callable,
    old_algo: Callable,
    new_kwargs: Optional[Dict] = None,
    old_kwargs: Optional[Dict] = None,
    name_new: str = "new",
    name_old: str = "old",
    feasibility_tol: float = 1e-8,
    x_tol: float = 1e-6,
    print_report: bool = True,
) -> Tuple[AlgoResult, AlgoResult, Dict]:
    """Run two algorithms and compare their outputs."""
    new_kwargs = new_kwargs or {}
    old_kwargs = old_kwargs or {}

    res_new = normalize_algo_output(name_new, new_algo(a, r_players, **new_kwargs))
    res_old = normalize_algo_output(name_old, old_algo(a, r_players, **old_kwargs))

    rep: Dict[str, Any] = {
        "k_new": res_new.k.tolist() if res_new.k is not None else None,
        "k_old": res_old.k.tolist() if res_old.k is not None else None,
        "bc_new": res_new.bc_penalty, "bc_old": res_old.bc_penalty,
        "d_rows_new": res_new.d_rows, "d_rows_old": res_old.d_rows,
        "feas_new": feasibility_report(res_new.X, r_players, tol=feasibility_tol),
        "feas_old": feasibility_report(res_old.X, r_players, tol=feasibility_tol),
        "X_cmp": compare_X(res_new.X, res_old.X),
    }
    if res_new.k is not None and res_old.k is not None and res_new.k.shape == res_old.k.shape:
        rep["k_match"] = bool(np.all(res_new.k == res_old.k))
    rep["X_close"] = bool(rep["X_cmp"].get("shape_match", 0.0) == 1.0 and rep["X_cmp"]["max_abs"] <= x_tol)

    if print_report:
        print("=== Algorithm comparison ===")
        for key in ["k_match", "k_new", "k_old", "bc_new", "bc_old", "d_rows_new", "d_rows_old"]:
            if key in rep:
                print(f"{key}: {rep[key]}")
        print("Feasibility (new):", rep["feas_new"])
        print("Feasibility (old):", rep["feas_old"])
        print("X diff:", rep["X_cmp"])
        print(f"X_close (tol={x_tol}): {rep['X_close']}")

    return res_new, res_old, rep
