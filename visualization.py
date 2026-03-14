from __future__ import annotations

import os
import shutil
from typing import Any, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.patches import Rectangle

from gmu_types import MarginalsResult, OuterSnapshot
from single_player import compute_player_rewards


# ===========================================================================
# BR history plots (for solve_gmu_best_response with track_history=True)
# ===========================================================================

def plot_player_history(history: np.ndarray, player: int, projects: Optional[List[int]] = None) -> None:
    """Plot allocation x[player, i] over BR iterations for selected projects."""
    steps, m, n = history.shape
    if not (0 <= player < m):
        raise ValueError("player out of range.")
    if projects is None:
        projects = list(range(n))
    for i in projects:
        plt.plot(np.arange(steps), history[:, player, i], label=f"proj {i}")
    plt.xlabel("step")
    plt.ylabel(f"x[player={player}, i]")
    plt.title(f"Allocation history for player {player}")
    plt.legend()
    plt.show()


def plot_totals_history(history: np.ndarray, projects: Optional[List[int]] = None) -> None:
    """Plot total resources on each project over BR iterations."""
    steps, m, n = history.shape
    totals = history.sum(axis=1)
    if projects is None:
        projects = list(range(n))
    for i in projects:
        plt.plot(np.arange(steps), totals[:, i], label=f"proj {i}")
    plt.xlabel("step")
    plt.ylabel("total on project i")
    plt.title("Total resources per project over time")
    plt.legend()
    plt.show()


def plot_marginal_rates_active_only(
    history: np.ndarray,
    A: np.ndarray,
    player: int,
    eps: float = 1e-12,
) -> None:
    """Plot marginal rates for active projects of a given player over BR iterations."""
    steps, m_players, n_projects = history.shape
    A = np.asarray(A, dtype=float)
    T = history.sum(axis=1)
    denom = (1.0 + T)[:, None, :]
    MR = A[None, :, :] * (denom - history) / (denom ** 2)
    x_player = history[:, player, :]
    mr_player = MR[:, player, :]
    ever_active = np.any(x_player > eps, axis=0)
    active_projects = np.where(ever_active)[0]
    if active_projects.size == 0:
        print(f"No active projects for player {player}.")
        return
    for i in active_projects:
        y = mr_player[:, i].copy()
        y[x_player[:, i] <= eps] = np.nan
        plt.plot(np.arange(steps), y, label=f"proj {i}")
    plt.xlabel("step")
    plt.ylabel(r"$\partial F^j/\partial x_i^j$")
    plt.title(f"Marginal rates (active only) for player {player}")
    plt.legend()
    plt.show()


# ===========================================================================
# Restricted GMU solver history plots
# ===========================================================================

def plot_c_history(solver, players: Optional[List[int]] = None, zones: Optional[List[int]] = None) -> None:
    """Plot c_{j,s} over solver iterations for selected players and zones."""
    if not solver.history:
        raise ValueError("No history recorded.")
    if players is None:
        players = list(range(solver.m))
    if zones is None:
        zones = list(range(solver.S))
    iters = [h.iter_idx for h in solver.history]
    for j in players:
        plt.figure()
        for s in zones:
            plt.plot(iters, [h.c_rows[j, s] for h in solver.history], label=f"zone {s+1}")
        plt.xlabel("iteration")
        plt.ylabel(f"c_{{j,s}} player {j+1}")
        plt.title(f"Marginal rates over iterations (player {j+1})")
        plt.legend()
        plt.tight_layout()


def plot_r_history(solver, players: Optional[List[int]] = None, zones: Optional[List[int]] = None) -> None:
    """Plot r_{j,s} over solver iterations for selected players and zones."""
    if not solver.history:
        raise ValueError("No history recorded.")
    if players is None:
        players = list(range(solver.m))
    if zones is None:
        zones = list(range(solver.S))
    iters = [h.iter_idx for h in solver.history]
    for j in players:
        plt.figure()
        for s in zones:
            plt.plot(iters, [h.r_zone[j, s] for h in solver.history], label=f"zone {s+1}")
        plt.xlabel("iteration")
        plt.ylabel(f"r_{{j,s}} player {j+1}")
        plt.title(f"Resources over iterations (player {j+1})")
        plt.legend()
        plt.tight_layout()


def plot_R_C_history(solver, zones: Optional[List[int]] = None) -> None:
    """Plot column sums R_s and marginal rates C_s over solver iterations."""
    if not solver.history:
        raise ValueError("No history recorded.")
    if zones is None:
        zones = list(range(solver.S))
    iters = [h.iter_idx for h in solver.history]
    plt.figure()
    for s in zones:
        plt.plot(iters, [h.R_cols[s] for h in solver.history], label=f"R_{s+1}")
    plt.xlabel("iteration")
    plt.ylabel("R_s")
    plt.title("Column resources R_s over iterations")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for s in zones:
        plt.plot(iters, [h.C_cols[s] for h in solver.history], label=f"C_{s+1}")
    plt.xlabel("iteration")
    plt.ylabel("C_s")
    plt.title("Cumulative marginal rates C_s over iterations")
    plt.legend()
    plt.tight_layout()


# ===========================================================================
# Visualization helpers for the monotone-projects animation
# ===========================================================================

def _unicode_subscript(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def zones_from_k(k: list, n: int) -> list:
    """
    Build contiguous zone intervals [i0, i1) with constant active-player sets.
    Returns list of (i0, i1, players_tuple).
    """
    m = len(k)
    k = [int(min(max(0, kj), n)) for kj in k]
    active_sets = [tuple(j for j in range(m) if k[j] > i) for i in range(n)]
    zones = []
    i0 = 0
    while i0 < n:
        s = active_sets[i0]
        i1 = i0 + 1
        while i1 < n and active_sets[i1] == s:
            i1 += 1
        zones.append((i0, i1, s))
        i0 = i1
    return zones


def _soft_player_colors(m: int, alpha: float = 0.25) -> list:
    """Soft pastel per-player colors (one per player from tab10)."""
    cmap = plt.get_cmap("tab10")
    cols = []
    for j in range(m):
        r, g, b, _ = cmap(j % 10)
        mix = 0.65
        cols.append((mix * r + (1 - mix), mix * g + (1 - mix), mix * b + (1 - mix), alpha))
    return cols


def _player_marginal_C(a: np.ndarray, X: np.ndarray, active_tol: float = 1e-10) -> np.ndarray:
    """
    Player shadow rate c_j = mean of d_{j,i} = a_i*(1+L_{-j,i})/(1+L_i)^2 over active projects.
    """
    a = np.asarray(a, float)
    X = np.asarray(X, float)
    m, n = X.shape
    L = X.sum(axis=0)
    denom = (1.0 + L) ** 2
    c = np.full(m, np.nan, dtype=float)
    for j in range(m):
        active = np.where(X[j] > active_tol)[0]
        if active.size == 0:
            continue
        c[j] = float(np.mean(a[active] * (1.0 + L[active] - X[j, active]) / denom[active]))
    return c


def _infer_k_from_X(X: np.ndarray, active_tol: float = 1e-10) -> list:
    X = np.asarray(X, float)
    m, n = X.shape
    k = []
    for j in range(m):
        idx = np.where(X[j] > active_tol)[0]
        k.append(int(idx.max() + 1) if idx.size else 0)
    return k


def _axis_from_display_for_rates(
    display_snaps: list,
    a: np.ndarray,
    active_tol: float,
    clip_q: float = 0.95,
) -> Tuple[Tuple[float, float], float]:
    """Compute fixed x-axis limits for the rate line across all display frames."""
    a = np.asarray(a, float)
    vals: list = []
    for snap in display_snaps:
        X = np.asarray(snap.X, float)
        m, n = X.shape
        c = _player_marginal_C(a, X, active_tol=active_tol)
        vals.extend([float(v) for v in c if np.isfinite(v)])
        j = getattr(snap, "player", None)
        if j is None:
            continue
        j = int(j)
        if not (0 <= j < m):
            continue
        active_idx = np.where(X[j] > active_tol)[0]
        last = int(active_idx.max()) if active_idx.size else -1
        cand = np.arange(last + 1, n)
        if cand.size == 0:
            continue
        L_minus = X.sum(axis=0) - X[j]
        vals.extend([float(v) for v in a[cand] / (1.0 + L_minus[cand]) if np.isfinite(v)])

    if not vals:
        return (0.0, 1.0), 1.0
    arr = np.asarray([v for v in vals if np.isfinite(v)], float)
    cap = float(np.quantile(arr, clip_q))
    lo = float(np.min(arr))
    pad = 0.10 * (cap - lo + 1e-12)
    return (lo - pad, cap + pad), cap


def _stagger_offsets(xs: list, dx: float) -> list:
    if not xs:
        return []
    order = np.argsort(xs)
    offs = np.zeros(len(xs), dtype=float)
    run = [order[0]]
    for ki in order[1:]:
        if abs(xs[ki] - xs[run[-1]]) < 0.04 * (max(xs) - min(xs) + 1e-12):
            run.append(ki)
        else:
            if len(run) > 1:
                grid = np.linspace(-dx, dx, len(run))
                for t, idx in enumerate(run):
                    offs[idx] = grid[t]
            run = [ki]
    if len(run) > 1:
        grid = np.linspace(-dx, dx, len(run))
        for t, idx in enumerate(run):
            offs[idx] = grid[t]
    return offs.tolist()


def _draw_rate_line(
    axr,
    *,
    a: np.ndarray,
    X: np.ndarray,
    active_tol: float,
    player_colors: list,
    considered_player: Optional[int],
    tested_project: Optional[int],
    post_project: Optional[int],
    post_rate: Optional[float],
    xlim: Tuple[float, float],
    cap: float,
    title: str = "",
    font_size: int = 12,
    marker_scale: float = 2.0,
) -> None:
    """Draw the rate line: player c_j (dots) and candidate entry rates (triangles)."""
    a = np.asarray(a, float)
    X = np.asarray(X, float)
    m, n = X.shape

    axr.clear()
    axr.set_xlim(*xlim)
    axr.set_ylim(-0.9, 0.9)
    axr.set_yticks([])
    axr.tick_params(axis="x", labelsize=font_size)
    for sp in ["left", "right", "top"]:
        axr.spines[sp].set_visible(False)
    axr.axhline(0.0, linewidth=1.2)

    if title:
        axr.text(0.5, 0.98, title, transform=axr.transAxes, ha="center", va="top", clip_on=True, zorder=10)

    C_players = _player_marginal_C(a, X, active_tol=active_tol)
    y_players = np.linspace(-0.55, -0.20, m) if m > 1 else np.array([-0.35])
    base_s = 60 * marker_scale
    focus_s = 105 * marker_scale

    xs = [float(cj) if np.isfinite(cj) else np.nan for cj in C_players]
    dx = 0.015 * (xlim[1] - xlim[0] + 1e-12)
    finite_xs = [x for x in xs if np.isfinite(x)]
    xoff_finite = _stagger_offsets(finite_xs, dx=dx)
    xoff_full = [0.0] * m
    ptr = 0
    for j in range(m):
        if np.isfinite(xs[j]):
            xoff_full[j] = xoff_finite[ptr]
            ptr += 1

    for j in range(m):
        cj = C_players[j]
        if not np.isfinite(cj):
            continue
        col = player_colors[j]
        is_focus = (considered_player is not None and j == considered_player)
        axr.scatter([cj], [y_players[j]], s=(focus_s if is_focus else base_s), marker="o",
                    color=col[:3], edgecolor="black", linewidth=(2.4 if is_focus else 1.3), zorder=4)
        axr.text(cj + xoff_full[j], y_players[j] - 0.12, f"P{j+1}",
                 ha="center", va="top", fontsize=font_size)

    if considered_player is not None and 0 <= considered_player < m:
        j = int(considered_player)
        active_idx = np.where(X[j] > active_tol)[0]
        last = int(active_idx.max()) if active_idx.size else -1
        cand = np.arange(last + 1, n)

        cj = float(C_players[j]) if np.isfinite(C_players[j]) else np.nan
        if np.isfinite(cj):
            axr.axvline(cj, ymin=0.08, ymax=0.90, linewidth=2.2, linestyle="--", color="black", zorder=1)

        if cand.size:
            L_minus = X.sum(axis=0) - X[j]
            qj = a[cand] / (1.0 + L_minus[cand])
            tri_s = 80 * marker_scale
            tri_s_test = 120 * marker_scale
            y0 = 0.08
            jitter = np.array([0.00, 0.12, 0.06, 0.18, 0.10, 0.24])

            t_test = None
            if tested_project is not None:
                i_star = int(tested_project)
                if cand.size and cand[0] <= i_star <= cand[-1]:
                    t_test = int(i_star - cand[0])

            if t_test is not None:
                q_test = float(qj[t_test])
                q_test_clip = min(q_test, cap)
                axr.axvline(q_test_clip, ymin=0.25, ymax=0.90, linewidth=2.4, linestyle=":", color="red", zorder=1)
                axr.text(q_test_clip, 0.4, f"testing P{cand[t_test]+1}" + ("→" if q_test > cap else ""),
                         ha="center", va="bottom",
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=1.0),
                         fontsize=font_size, color="red", zorder=5)

            for t, i in enumerate(cand):
                val = float(qj[t])
                clipped = min(val, cap)
                y = min(y0 + jitter[t % len(jitter)], 0.50)
                is_test = (t_test is not None and t == t_test)
                axr.scatter([clipped], [y], s=(tri_s_test if is_test else tri_s), marker="^",
                            color=("red" if is_test else "black"),
                            edgecolor="black", linewidth=(2.2 if is_test else 1.0), zorder=3)
                if not is_test:
                    axr.text(clipped, y + 0.12, f"P{i+1}" + ("→" if val > cap else ""),
                             ha="center", va="bottom", fontsize=font_size)

        if post_project is not None and post_rate is not None and np.isfinite(post_rate):
            x_post = min(float(post_rate), cap)
            axr.scatter([x_post], [0.38], s=115 * marker_scale, marker="^",
                        facecolors="none", edgecolors="black", linewidths=2.0, zorder=6)
            axr.text(x_post, 0.50, f"P{int(post_project)+1} (after)" + ("→" if float(post_rate) > cap else ""),
                     ha="center", va="bottom", fontsize=font_size,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=1.0), zorder=6)

    axr.set_xlabel("rates: player cⱼ (dots) vs candidate project rates (triangles)", fontsize=font_size)


def render_state_table(
    ax,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    X: np.ndarray,
    k: Optional[list] = None,
    bc_penalty: Optional[float] = None,
    step: Optional[int] = None,
    active_tol: float = 1e-10,
    show_numbers: bool = True,
    alpha: float = 0.25,
    font_size: int = 10,
    header_font_size: int = 11,
    phase: Optional[str] = None,
    player: Optional[int] = None,
    target_K: Optional[int] = None,
    rate_line: bool = False,
    rate_xlim: Tuple[float, float] = (0.0, 1.0),
    rate_cap: float = 1.0,
    considered_player: Optional[int] = None,
    rate_title: str = "",
    rate_tested_project: Optional[int] = None,
    rate_post_project: Optional[int] = None,
    rate_post_rate: Optional[float] = None,
    final_frame: bool = False,
) -> None:
    """
    Render a step-by-step state table for the GMU algorithm.

    Layout (axes coordinates):
      - Top row: project labels and a values
      - Zone header strip
      - Main grid: allocation per player per project (colored by player)
      - Right columns: player shadow rate c_j and reward u_j
      - Bottom rows: R per project, R per zone, C per zone
      - (Optional) Rate line inset at the very bottom
    """
    ax.clear()
    a = np.asarray(a, float)
    r_players = np.asarray(r_players, float).reshape(-1)
    X = np.asarray(X, float)
    m, n = X.shape

    if k is None:
        k = _infer_k_from_X(X, active_tol=active_tol)
    else:
        k = [int(v) for v in k]

    zones = zones_from_k(k, n)
    L = X.sum(axis=0)
    Cproj = a / (1.0 + L)
    zone_meta = [dict(i0=i0, i1=i1, players=p,
                      R=float(np.sum(L[i0:i1])),
                      C=float(np.mean(Cproj[i0:i1])) if i1 > i0 else 0.0)
                 for (i0, i1, p) in zones]

    C_players = _player_marginal_C(a, X, active_tol=active_tol)
    U_players, U_total = compute_player_rewards(a, X)
    player_cols = _soft_player_colors(m, alpha=alpha)
    datum_gray = (0.90, 0.90, 0.90, 1.0)

    action_col = (int(target_K) - 1) if (target_K is not None and 1 <= int(target_K) <= n) else \
                 max(0, min(int(max(k) - 1), n - 1))

    # --- Layout ---
    x_left = 0.03
    w_left = 0.19
    gap_lr = 0.045
    x_core = x_left + w_left + gap_lr

    gap_core_right = 0.015
    w_right = 0.14
    w_core = 0.96 - x_core - gap_core_right - w_right
    x_right = x_core + w_core + gap_core_right
    w_c = w_u = w_right / 2.0
    x_c = x_right
    x_u = x_right + w_c

    rate_y = 0.03
    rate_h = 0.18
    rate_gap = 0.03
    h_bottom = 0.19
    gap_cb = 0.03
    y_bottom = rate_y + rate_h + rate_gap
    h_core = 0.36
    y_core = y_bottom + h_bottom + gap_cb
    gap_ac = 0.05
    h_top = 0.09
    y_top = y_core + h_core + gap_ac
    h_sum = 0.055
    gap_sum = 0.008
    y_sum = y_core - h_sum - gap_sum
    header_h = 0.08
    y_zonehdr = y_core + h_core - header_h
    col_w = w_core / n

    # Title
    title = "GMU evolution"
    if step is not None:
        title += f" | step={step}"
    if bc_penalty is not None:
        title += f" | BC={bc_penalty:.3e}"
    ax.text(0.5, y_top + h_top + 0.02, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15)

    # Top table: project labels and a values
    w_top_title = 0.08
    x_top_title = x_core - w_top_title
    top_title_tbl = ax.table(cellText=[["Project"], ["a"]], cellLoc="center",
                              bbox=[x_top_title, y_top, w_top_title, h_top])
    top_title_tbl.auto_set_font_size(False)
    top_title_tbl.set_fontsize(header_font_size)
    for (rr, cc), cell in top_title_tbl.get_celld().items():
        cell.set_linewidth(1.0)
        if rr == 1:
            cell.set_facecolor(datum_gray)

    top_tbl = ax.table(
        cellText=[
            [f"P{_unicode_subscript(i+1)}" for i in range(n)],
            [f"{a[i]:.3g}" if show_numbers else "" for i in range(n)],
        ],
        cellLoc="center", bbox=[x_core, y_top, w_core, h_top],
    )
    top_tbl.auto_set_font_size(False)
    top_tbl.set_fontsize(font_size)
    for (rr, cc), cell in top_tbl.get_celld().items():
        cell.set_linewidth(1.0)
        if rr == 0:
            cell.set_fontsize(header_font_size)
        if rr == 1:
            cell.set_facecolor(datum_gray)

    # Zone header strip
    K_max = max(0, min(int(max(k)), n))
    ax.add_patch(Rectangle((x_core, y_zonehdr), w_core, header_h, fill=False, edgecolor="black", linewidth=1.2))
    z_id = 0
    for meta in zone_meta:
        i0, i1 = meta["i0"], min(meta["i1"], K_max)
        if i1 <= meta["i0"]:
            continue
        z_id += 1
        x0 = x_core + meta["i0"] * col_w
        w0 = (i1 - meta["i0"]) * col_w
        ax.add_patch(Rectangle((x0, y_zonehdr), w0, header_h, facecolor="white", edgecolor="black", linewidth=1.0))
        ax.text(x0 + w0 / 2, y_zonehdr + header_h / 2, f"Zone {z_id}",
                ha="center", va="center", fontsize=header_font_size)
    for i in range(K_max, n):
        ax.add_patch(Rectangle((x_core + i * col_w, y_zonehdr), col_w, header_h,
                                facecolor="white", edgecolor="black", linewidth=1.0))

    # Left header: [Player, r]
    ax.add_patch(Rectangle((x_left, y_zonehdr), w_left, header_h, fill=False, edgecolor="black", linewidth=1.2))
    w_col_left = w_left / 2.0
    for col_idx, label in enumerate(["Player", "r"]):
        x0 = x_left + col_idx * w_col_left
        ax.add_patch(Rectangle((x0, y_zonehdr), w_col_left, header_h, facecolor="white", edgecolor="black", linewidth=1.0))
        ax.text(x0 + w_col_left / 2, y_zonehdr + header_h / 2, label, ha="center", va="center", fontsize=header_font_size)

    left_tbl = ax.table(
        cellText=[[f"P{j+1}", f"{r_players[j]:.3g}"] for j in range(m)],
        cellLoc="center", bbox=[x_left, y_core, w_left, h_core - header_h],
    )
    left_tbl.auto_set_font_size(False)
    left_tbl.set_fontsize(font_size)
    for (rr, cc), cell in left_tbl.get_celld().items():
        cell.set_linewidth(1.0)
        col = player_cols[rr]
        cell.set_facecolor((col[0], col[1], col[2], alpha) if cc == 0 else datum_gray)

    sum_tbl = ax.table(cellText=[["Σ", f"{float(np.sum(r_players)):.3g}"]],
                       cellLoc="center", bbox=[x_left, y_sum, w_left, h_sum])
    sum_tbl.auto_set_font_size(False)
    sum_tbl.set_fontsize(font_size)
    for cell in sum_tbl.get_celld().values():
        cell.set_linewidth(1.0)
        cell.set_facecolor(datum_gray)

    # Right panel: [c, Reward]
    ax.add_patch(Rectangle((x_right, y_zonehdr), w_right, header_h, fill=False, edgecolor="black", linewidth=1.2))
    for x0, w0, lab in [(x_c, w_c, "c"), (x_u, w_u, "Reward")]:
        ax.add_patch(Rectangle((x0, y_zonehdr), w0, header_h, facecolor="white", edgecolor="black", linewidth=1.0))
        ax.text(x0 + w0 / 2, y_zonehdr + header_h / 2, lab, ha="center", va="center", fontsize=header_font_size)

    right_tbl = ax.table(
        cellText=[
            [("" if np.isnan(C_players[j]) else f"{C_players[j]:.3g}"),
             ("" if not np.isfinite(U_players[j]) else f"{U_players[j]:.4g}")]
            for j in range(m)
        ],
        cellLoc="center", bbox=[x_right, y_core, w_right, h_core - header_h],
    )
    right_tbl.auto_set_font_size(False)
    right_tbl.set_fontsize(font_size)
    for (rr, cc), cell in right_tbl.get_celld().items():
        cell.set_linewidth(1.0)
        col = player_cols[rr]
        cell.set_facecolor((col[0], col[1], col[2], alpha))
        if final_frame:
            txt = cell.get_text().get_text()
            if txt.strip():
                cell.get_text().set_fontweight("bold")

    sum_right_tbl = ax.table(cellText=[["Σ", f"{U_total:.4g}"]],
                              cellLoc="center", bbox=[x_right, y_sum, w_right, h_sum])
    sum_right_tbl.auto_set_font_size(False)
    sum_right_tbl.set_fontsize(font_size)
    for cell in sum_right_tbl.get_celld().values():
        cell.set_linewidth(1.0)
        cell.set_facecolor((0.92, 0.92, 0.92, 1.0))

    # Core allocation grid
    core_tbl = ax.table(
        cellText=[[f"{X[j,i]:.3g}" if (X[j,i] > active_tol and show_numbers) else "" for i in range(n)] for j in range(m)],
        cellColours=[[player_cols[j] if X[j,i] > active_tol else (1,1,1,1) for i in range(n)] for j in range(m)],
        cellLoc="center", bbox=[x_core, y_core, w_core, h_core - header_h],
    )
    core_tbl.auto_set_font_size(False)
    core_tbl.set_fontsize(font_size)
    for cell in core_tbl.get_celld().values():
        cell.set_linewidth(1.0)
    if final_frame:
        for cell in core_tbl.get_celld().values():
            if cell.get_text().get_text().strip():
                cell.get_text().set_fontweight("bold")

    if not final_frame and action_col is not None and 0 <= action_col < n:
        ax.add_patch(Rectangle((x_core + action_col * col_w, y_core), col_w, h_core - header_h,
                                fill=False, edgecolor="black", linewidth=3.2))

    # Bottom: R per project, R per zone, C per zone
    row_h = h_bottom / 3.0
    y_Rproj = y_bottom + 2.0 * row_h
    y_Rzone = y_bottom + 1.0 * row_h
    y_Czone = y_bottom
    ax.add_patch(Rectangle((x_core, y_bottom), w_core, h_bottom, fill=False, edgecolor="black", linewidth=1.2))
    for label, y_row in [("R", y_Rproj), ("R_zone", y_Rzone), ("C_zone", y_Czone)]:
        ax.text(x_core - 0.02, y_row + row_h / 2, label, ha="right", va="center", fontsize=header_font_size)

    fw = "bold" if final_frame else "normal"
    for i in range(n):
        x0 = x_core + i * col_w
        ax.add_patch(Rectangle((x0, y_Rproj), col_w, row_h, facecolor="white", edgecolor="black", linewidth=1.0))
        if show_numbers:
            ax.text(x0 + col_w / 2, y_Rproj + row_h / 2, f"{L[i]:.3g}", ha="center", va="center",
                    fontsize=font_size, fontweight=fw)

    for meta in zone_meta:
        i0, i1 = meta["i0"], meta["i1"]
        if len(meta["players"]) == 0:
            for i in range(i0, i1):
                x0 = x_core + i * col_w
                for y_row, val in [(y_Rzone, L[i]), (y_Czone, Cproj[i])]:
                    ax.add_patch(Rectangle((x0, y_row), col_w, row_h, facecolor="white", edgecolor="black", linewidth=1.0))
                    if show_numbers:
                        ax.text(x0 + col_w / 2, y_row + row_h / 2, f"{val:.3g}", ha="center", va="center",
                                fontsize=font_size, fontweight=fw)
        else:
            x0 = x_core + i0 * col_w
            w0 = (i1 - i0) * col_w
            for y_row, val in [(y_Rzone, meta["R"]), (y_Czone, meta["C"])]:
                ax.add_patch(Rectangle((x0, y_row), w0, row_h, facecolor="white", edgecolor="black", linewidth=1.0))
                if show_numbers:
                    ax.text(x0 + w0 / 2, y_row + row_h / 2, f"{val:.3g}", ha="center", va="center",
                            fontsize=font_size, fontweight=fw)

    # Rate line inset
    if rate_line:
        axr = ax.inset_axes([0.10, rate_y, 0.86, rate_h])
        _draw_rate_line(
            axr, a=a, X=X, active_tol=active_tol,
            player_colors=[(c[0], c[1], c[2], 0.95) for c in player_cols],
            considered_player=considered_player,
            tested_project=rate_tested_project,
            post_project=rate_post_project,
            post_rate=rate_post_rate,
            xlim=rate_xlim, cap=rate_cap, title=rate_title, font_size=10,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ===========================================================================
# Animation helpers
# ===========================================================================

def _compress_tighten(hist: list) -> list:
    """Group each (event + trailing tighten frames) into one stable frame."""
    frames = []
    i = 0
    while i < len(hist):
        s = hist[i]
        if getattr(s, "phase", "") == "tighten":
            frames.append(s)
            i += 1
            continue
        event_phase = getattr(s, "phase", "")
        event_step = getattr(s, "step", None)
        last = s
        j = i + 1
        while j < len(hist):
            t = hist[j]
            if getattr(t, "phase", "") != "tighten":
                break
            if event_step is not None and getattr(t, "step", None) != event_step:
                break
            last = t
            j += 1

        class _Proxy: ...
        p = _Proxy()
        for attr in ["X", "k", "bc_penalty", "step", "target_K", "player", "phase"]:
            if hasattr(last, attr):
                setattr(p, attr, getattr(last, attr))
        p.phase = event_phase
        frames.append(p)
        i = j
    return frames


def _build_display_frames(
    base_frames: list,
    *,
    active_tol: float,
    include_testing_frames: bool,
) -> list:
    """Build (mode, snap) display list with optional BEFORE/AFTER pairs and a FINAL frame."""

    def _is_accept(ph: str) -> bool:
        return ph.startswith(("expand_", "fallback_"))

    def _same_state(a, b, atol: float = 1e-12) -> bool:
        if getattr(a, "k", None) != getattr(b, "k", None):
            return False
        return np.allclose(np.asarray(a.X, float), np.asarray(b.X, float), rtol=0.0, atol=atol)

    display = [("after", base_frames[0])]

    if include_testing_frames:
        for t in range(1, len(base_frames)):
            prev, cur = base_frames[t - 1], base_frames[t]
            ph = getattr(cur, "phase", "")
            player = getattr(cur, "player", None)

            tested_project = None
            if player is not None:
                jj = int(player)
                Xprev = np.asarray(prev.X, float)
                idx = np.where(Xprev[jj] > active_tol)[0]
                last = int(idx.max()) if idx.size else -1
                tp = last + 1
                if 0 <= tp < Xprev.shape[1]:
                    tested_project = tp

            class _Before: ...
            b = _Before()
            b.X, b.k = prev.X, getattr(prev, "k", None)
            b.bc_penalty, b.step = getattr(prev, "bc_penalty", None), getattr(cur, "step", t)
            b.phase, b.player, b.tested_project = ph, player, tested_project
            display.append(("before", b))

            if _is_accept(ph):
                cur.tested_project = tested_project
                display.append(("after", cur))
    else:
        last_kept = base_frames[0]
        for t in range(1, len(base_frames)):
            cur = base_frames[t]
            if _same_state(last_kept, cur):
                continue
            display.append(("after", cur))
            last_kept = cur

    _last_mode, last_snap = display[-1]

    class _Final: ...
    fin = _Final()
    fin.X, fin.k = last_snap.X, getattr(last_snap, "k", None)
    fin.bc_penalty, fin.step = getattr(last_snap, "bc_penalty", None), getattr(last_snap, "step", None)
    fin.phase, fin.player, fin.tested_project = "final", None, None
    display.append(("final", fin))

    return display


# ===========================================================================
# Public animators
# ===========================================================================

def animate_outer_history(
    outer_hist: list,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    active_tol: float = 1e-10,
    interval_ms: int = 800,
    show_numbers: bool = True,
    figsize: Tuple[float, float] = (14, 6.5),
    alpha: float = 0.25,
    font_size: int = 10,
    header_font_size: int = 11,
    compress_tighten: bool = True,
) -> "HTML":
    """
    Animate the outer algorithm history (no rate line).
    Returns an IPython HTML object for display in Jupyter.
    """
    hist = [s for s in outer_hist if getattr(s, "X", None) is not None]
    if not hist:
        raise ValueError("No frames with X in outer_hist. Run with store_X_in_history=True.")

    frames = _compress_tighten(hist) if compress_tighten else hist
    fig, ax = plt.subplots(figsize=figsize)

    def _update(t: int):
        snap = frames[t]
        render_state_table(
            ax, a=a, r_players=r_players, X=snap.X,
            k=getattr(snap, "k", None), bc_penalty=None, step=getattr(snap, "step", t),
            active_tol=active_tol, show_numbers=show_numbers, alpha=alpha,
            font_size=font_size, header_font_size=header_font_size,
            phase=getattr(snap, "phase", None), player=getattr(snap, "player", None),
            target_K=getattr(snap, "target_K", None), rate_line=False,
        )
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=interval_ms, blit=False, repeat=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def animate_outer_history_rates(
    outer_hist: list,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    active_tol: float = 1e-10,
    interval_ms: int = 700,
    show_numbers: bool = True,
    alpha: float = 0.25,
    font_size: int = 10,
    header_font_size: int = 11,
    compress_tighten: bool = True,
    clip_q: float = 0.95,
    figsize: Tuple[float, float] = (14, 6.5),
    rate_line: bool = False,
) -> "HTML":
    """
    Animate the outer history with an optional rate line.

    rate_line=False: show AFTER states only (compact, one frame per step).
    rate_line=True : show BEFORE/AFTER pairs to illustrate the rate comparison.
    """
    a = np.asarray(a, float)
    hist = [s for s in outer_hist if getattr(s, "X", None) is not None]
    if not hist:
        raise ValueError("No frames with X in outer_hist.")

    base_frames = _compress_tighten(hist) if compress_tighten else hist
    display = _build_display_frames(base_frames, active_tol=active_tol, include_testing_frames=rate_line)

    rate_xlim, rate_cap = _axis_from_display_for_rates(
        [snap for _, snap in display], a=a, active_tol=active_tol, clip_q=clip_q,
    )

    fig, ax = plt.subplots(figsize=figsize)

    def _post_marginal(X_: np.ndarray, j: int, i: int) -> float:
        L_ = X_.sum(axis=0)
        return float(a[i] * (1.0 + L_[i] - X_[j, i]) / (1.0 + L_[i]) ** 2)

    def _update(idx: int):
        mode, snap = display[idx]
        X = np.asarray(snap.X, float)
        is_final = (mode == "final")
        j = getattr(snap, "player", None)
        j_int = int(j) if (j is not None and not is_final) else None
        tested = getattr(snap, "tested_project", None)
        tested_for_plot = tested if mode == "before" else None

        post_project = post_rate = None
        if rate_line and mode == "after" and j_int is not None and tested is not None:
            if 0 <= tested < X.shape[1]:
                post_project = tested
                post_rate = _post_marginal(X, j_int, tested)

        if is_final:
            rate_title = "FINAL"
        elif rate_line and j_int is not None and tested is not None:
            rate_title = ("TESTING" if mode == "before" else "AFTER") + f" | P{j_int+1} → project {tested+1}"
        elif j_int is not None:
            rate_title = f"{mode.upper()} | P{j_int+1}"
        else:
            rate_title = mode.upper()

        render_state_table(
            ax, a=a, r_players=r_players, X=X,
            k=getattr(snap, "k", None), bc_penalty=None, step=getattr(snap, "step", idx),
            active_tol=active_tol, show_numbers=show_numbers, alpha=alpha,
            font_size=font_size, header_font_size=header_font_size,
            phase=getattr(snap, "phase", None), player=j_int, target_K=None,
            rate_line=rate_line, rate_xlim=rate_xlim, rate_cap=rate_cap,
            considered_player=(None if is_final else j_int), rate_title=rate_title,
            rate_tested_project=(None if (is_final or not rate_line) else tested_for_plot),
            rate_post_project=(None if (is_final or not rate_line) else post_project),
            rate_post_rate=(None if (is_final or not rate_line) else post_rate),
            final_frame=is_final,
        )
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(display), interval=interval_ms, blit=False, repeat=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


# ===========================================================================
# Export to MP4 / PNG frames
# ===========================================================================

def save_outer_history_mp4(
    outer_hist: list,
    out_mp4: str,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    active_tol: float = 1e-10,
    show_numbers: bool = True,
    alpha: float = 0.18,
    font_size: int = 10,
    header_font_size: int = 11,
    figsize: Tuple[float, float] = (14, 6.5),
    compress_tighten: bool = True,
    fps: int = 6,
    dpi: int = 180,
    codec: str = "libx264",
) -> None:
    """Save outer history animation as MP4. Requires ffmpeg on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install via `conda install -c conda-forge ffmpeg`.")

    hist = [s for s in outer_hist if getattr(s, "X", None) is not None]
    if not hist:
        raise ValueError("No frames with X in outer_hist.")

    frames = _compress_tighten(hist) if compress_tighten else hist
    fig, ax = plt.subplots(figsize=figsize)

    def _update(t):
        snap = frames[t]
        render_state_table(
            ax, a=a, r_players=r_players, X=snap.X,
            k=getattr(snap, "k", None), bc_penalty=getattr(snap, "bc_penalty", None),
            step=getattr(snap, "step", t), active_tol=active_tol, show_numbers=show_numbers,
            alpha=alpha, font_size=font_size, header_font_size=header_font_size,
            phase=getattr(snap, "phase", None), player=getattr(snap, "player", None),
            target_K=getattr(snap, "target_K", None),
        )
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec=codec, bitrate=2000)
    ani.save(out_mp4, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved MP4: {out_mp4}  |  frames={len(frames)}  fps={fps}")


def save_outer_history_rates_frames(
    outer_hist: list,
    out_dir: str,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    active_tol: float = 1e-10,
    show_numbers: bool = True,
    alpha: float = 0.5,
    font_size: int = 10,
    header_font_size: int = 11,
    compress_tighten: bool = True,
    clip_q: float = 0.95,
    figsize: Tuple[float, float] = (14, 6.5),
    dpi: int = 220,
    prefix: str = "frame",
    ext: str = "png",
    rate_line: bool = False,
) -> list:
    """
    Save individual PNG frames (for Beamer/slides).

    rate_line=False: AFTER-only frames + FINAL.
    rate_line=True : BEFORE/AFTER pairs + FINAL.
    """
    a = np.asarray(a, float)
    os.makedirs(out_dir, exist_ok=True)

    hist = [s for s in outer_hist if getattr(s, "X", None) is not None]
    if not hist:
        raise ValueError("No frames with X in outer_hist.")

    base_frames = _compress_tighten(hist) if compress_tighten else hist
    display = _build_display_frames(base_frames, active_tol=active_tol, include_testing_frames=rate_line)

    rate_xlim, rate_cap = _axis_from_display_for_rates(
        [snap for _, snap in display], a=a, active_tol=active_tol, clip_q=clip_q,
    )

    def _post_marginal(X_: np.ndarray, j: int, i: int) -> float:
        L_ = X_.sum(axis=0)
        return float(a[i] * (1.0 + L_[i] - X_[j, i]) / (1.0 + L_[i]) ** 2)

    fig, ax = plt.subplots(figsize=figsize)
    saved_paths: list = []

    for frame_idx, (mode, snap) in enumerate(display):
        X = np.asarray(snap.X, float)
        is_final = (mode == "final")
        j = getattr(snap, "player", None)
        j_int = int(j) if (j is not None and not is_final) else None
        tested = getattr(snap, "tested_project", None)
        tested_for_plot = tested if mode == "before" else None

        post_project = post_rate = None
        if rate_line and mode == "after" and j_int is not None and tested is not None:
            if 0 <= tested < X.shape[1]:
                post_project = tested
                post_rate = _post_marginal(X, j_int, tested)

        if is_final:
            rate_title = "FINAL"
        elif rate_line and j_int is not None and tested is not None:
            rate_title = ("TESTING" if mode == "before" else "AFTER") + f" | P{j_int+1} → project {tested+1}"
        elif j_int is not None:
            rate_title = f"{mode.upper()} | P{j_int+1}"
        else:
            rate_title = mode.upper()

        render_state_table(
            ax, a=a, r_players=r_players, X=X,
            k=getattr(snap, "k", None), bc_penalty=None, step=getattr(snap, "step", frame_idx),
            active_tol=active_tol, show_numbers=show_numbers, alpha=alpha,
            font_size=font_size, header_font_size=header_font_size,
            phase=getattr(snap, "phase", None), player=j_int, target_K=None,
            rate_line=rate_line, rate_xlim=rate_xlim, rate_cap=rate_cap,
            considered_player=(None if is_final else j_int), rate_title=rate_title,
            rate_tested_project=(None if (is_final or not rate_line) else tested_for_plot),
            rate_post_project=(None if (is_final or not rate_line) else post_project),
            rate_post_rate=(None if (is_final or not rate_line) else post_rate),
            final_frame=is_final,
        )
        path = os.path.join(out_dir, f"{prefix}_{frame_idx:04d}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        saved_paths.append(path)

    plt.close(fig)
    print(f"Saved {len(saved_paths)} frames to: {out_dir}")
    return saved_paths


def save_outer_history_rates_mp4(
    outer_hist: list,
    out_mp4: str,
    *,
    a: np.ndarray,
    r_players: np.ndarray,
    active_tol: float = 1e-10,
    show_numbers: bool = True,
    alpha: float = 0.5,
    font_size: int = 10,
    header_font_size: int = 12,
    compress_tighten: bool = True,
    clip_q: float = 0.90,
    figsize: Tuple[float, float] = (12, 8),
    fps: int = 6,
    dpi: int = 200,
    codec: str = "libx264",
) -> None:
    """Save the rates animation as MP4 (BEFORE/AFTER pairs with rate line). Requires ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install via `conda install -c conda-forge ffmpeg`.")

    a = np.asarray(a, float)
    hist = [s for s in outer_hist if getattr(s, "X", None) is not None]
    if not hist:
        raise ValueError("No frames with X in outer_hist.")

    base_frames = _compress_tighten(hist) if compress_tighten else hist
    display = _build_display_frames(base_frames, active_tol=active_tol, include_testing_frames=True)

    rate_xlim, rate_cap = _axis_from_display_for_rates(
        [snap for _, snap in display], a=a, active_tol=active_tol, clip_q=clip_q,
    )

    def _post_marginal(X_: np.ndarray, j: int, i: int) -> float:
        L_ = X_.sum(axis=0)
        return float(a[i] * (1.0 + L_[i] - X_[j, i]) / (1.0 + L_[i]) ** 2)

    fig, ax = plt.subplots(figsize=figsize)

    def _update(frame_idx: int):
        mode, snap = display[frame_idx]
        X = np.asarray(snap.X, float)
        j = getattr(snap, "player", None)
        j_int = int(j) if j is not None else None
        tested = getattr(snap, "tested_project", None)
        tested_for_plot = tested if mode == "before" else None

        post_project = post_rate = None
        if mode == "after" and j_int is not None and tested is not None:
            if 0 <= tested < X.shape[1]:
                post_project = tested
                post_rate = _post_marginal(X, j_int, tested)

        if j_int is not None and tested is not None:
            rate_title = ("TESTING" if mode == "before" else "AFTER") + f" | P{j_int+1} → project {tested+1}"
        elif j_int is not None:
            rate_title = f"{mode.upper()} | P{j_int+1}"
        else:
            rate_title = mode.upper()

        render_state_table(
            ax, a=a, r_players=r_players, X=X,
            k=getattr(snap, "k", None), bc_penalty=getattr(snap, "bc_penalty", None),
            step=getattr(snap, "step", frame_idx), active_tol=active_tol,
            show_numbers=show_numbers, alpha=alpha, font_size=font_size, header_font_size=header_font_size,
            player=j_int, rate_line=False, rate_xlim=rate_xlim, rate_cap=rate_cap,
            considered_player=j_int, rate_title=rate_title,
            rate_tested_project=tested_for_plot, rate_post_project=post_project, rate_post_rate=post_rate,
        )
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(display), interval=1000 / fps, blit=False, repeat=False)
    writer = animation.FFMpegWriter(fps=fps, codec=codec, bitrate=2000)
    ani.save(out_mp4, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved MP4: {out_mp4} | frames={len(display)} | fps={fps} | codec={codec}")
