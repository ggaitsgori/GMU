from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SinglePlayerSolutionMU:
    k_star: int       # number of active projects
    c: float          # optimal marginal rate
    x: np.ndarray     # optimal allocation (sums to R)
    v: float          # optimal objective value


@dataclass
class GameSolutionBR:
    X: np.ndarray
    utilities: np.ndarray
    iters: int
    converged: bool
    history: Optional[np.ndarray] = None
    history_meta: Optional[dict] = None


@dataclass
class TwoPlayerOneZoneSolution:
    C: float
    c1: float
    c2: float
    x1: np.ndarray
    x2: np.ndarray
    L: np.ndarray


@dataclass
class TwoZoneSolutionForK:
    k2: int
    k1: int
    x1: np.ndarray
    x2: np.ndarray
    x_star: float
    c1: float
    c2: float


@dataclass
class TwoPlayerFullSolution:
    k2: int
    k1: int
    x1: np.ndarray
    x2: np.ndarray
    c1: float
    c2: float
    x_star: float


@dataclass
class NECheckResult:
    is_ne: bool
    c1: Optional[float]
    c2: Optional[float]
    details: dict


@dataclass
class ZoneSpec:
    project_indices: List[int]
    active_players: List[int]


@dataclass
class RestrictedGMUSpec:
    a: np.ndarray
    total_resources: np.ndarray
    zones: List[ZoneSpec]


@dataclass
class HistoryEntry:
    iter_idx: int
    r_zone: np.ndarray
    C_cols: np.ndarray
    c_rows: np.ndarray
    R_cols: np.ndarray


@dataclass
class ZoneSolution:
    C: float
    c_per_player: Dict[int, float]


@dataclass
class MarginalsResult:
    C_cols: np.ndarray
    c_rows: np.ndarray
    r_rows: np.ndarray
    disruptions: np.ndarray
    global_disruption: float


@dataclass
class GlobalFrame:
    step_idx: int
    k: List[int]
    zones: List[ZoneSpec]
    entry: HistoryEntry


@dataclass
class OuterSnapshot:
    step: int
    phase: str
    player: Optional[int]
    target_K: Optional[int]
    k: List[int]
    stopped: List[bool]
    bc_penalty: float
    bc_by_player: np.ndarray
    d_rows: float
    invested_target: Optional[float] = None
    note: str = ""
    X: Optional[np.ndarray] = None
    solver_last_entry: Any = None
    frame_id: int = 0


@dataclass
class AlgoResult:
    name: str
    k: Optional[np.ndarray]
    X: np.ndarray
    bc_penalty: Optional[float]
    d_rows: Optional[float]
    solver: Any = None
    marg: Any = None
    history: Any = None
    raw: Any = None
