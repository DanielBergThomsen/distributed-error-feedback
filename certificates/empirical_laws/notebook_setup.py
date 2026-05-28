"""Shared setup and helpers for the empirical-law notebooks."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import warnings

import numpy as np

root = Path(__file__).resolve().parents[2]  # repo root
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Shared solver settings for Lyapunov feasibility checks.
MOSEK_SINGLE_THREAD_SOLVE_KWARGS = {
    "mosek_params": {"MSK_IPAR_NUM_THREADS": 1},
}

MOSEK_STRICT_SOLVE_KWARGS = {
    "mosek_params": {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    },
}

SDPA_HIGH_PRECISION_SOLVE_KWARGS = {
    "print": "no",
    "numThreads": 1,
    "epsilonStar": 1e-11,
    "epsilonDash": 1e-11,
    "mpfPrecision": 4096,
    "maxIteration": 2000,
    "lambdaStar": 0.25,
    "gammaStar": 0.4,
    "omegaStar": 1.25,
    "betaStar": 0.1,
    "betaBar": 0.2,
}


ROOT_IMAG_TOL = 1e-7
ROOT_RANGE_TOL = 1e-8
ROOT_RESIDUAL_TOL = 1e-6


# SDPA/CVXPY can return a feasible-looking result while emitting known numerical
# warning text. Treat matching warning patterns as non-clean by default.
SDPA_WARNING_PATTERNS = [
    "cannot cholesky decomposition",
    "Could you try with smaller gammaStar?",
    "cannot move: step length is too short",
    "Step length is too small",
    "Solution may be inaccurate",
    "ARPACK error",
    "RuntimeWarning: Python recalculation of primal and/or dual feasibility error failed",
]


def scaled_problem_data_for_case(L_tuple, kappa_tuple):
    """Return max-L normalized problem data and metadata for one case."""
    raw_Ls = np.array(L_tuple, dtype=float)
    scale_down = float(np.max(raw_Ls))
    if not np.isfinite(scale_down) or scale_down <= 0.0:
        raise ValueError(f"Invalid max-L scale_down={scale_down!r} for L_tuple={L_tuple!r}")
    Ls = raw_Ls / scale_down
    mus = Ls / np.array(kappa_tuple, dtype=float)
    return {
        "scale_down": scale_down,
        "Ls": Ls,
        "mus": mus,
    }


def empirical_eta_star(eps, Ls, mus, n_workers):
    """Empirical Law 4.1 step size for EF/EF21."""
    return (2 * n_workers / np.sum(Ls + mus)) * ((1 - np.sqrt(eps)) / (1 + np.sqrt(eps)))


def rho_opt_n2(eps, Ls, mus, *, range_tol=ROOT_RANGE_TOL, residual_tol=ROOT_RESIDUAL_TOL):
    """Largest valid real root from Empirical Law 4.3 for the n=2 cubic rate."""
    Ls = np.asarray(Ls, dtype=float)
    mus = np.asarray(mus, dtype=float)
    if Ls.shape != (2,) or mus.shape != (2,):
        raise ValueError("rho_opt_n2 is only defined for n=2.")

    s = np.sqrt(float(eps))
    r_s = (1 - s) ** 2 / (1 + s)
    sigma = Ls + mus
    delta = Ls - mus
    s1, s2 = sigma
    d1, d2 = delta

    k1 = (d2**2 * s1 + d1**2 * s2) / (s1 * s2 * (s1 + s2))
    k2 = (d1 + d2) ** 2 / (s1 + s2) ** 2
    coeffs = np.array([
        1.0,
        -(s * (2 + s) + r_s * (s * k1 + k2)),
        s**2 * (1 + 2 * s + r_s * (k1 + s * k2)),
        -(s**4),
    ], dtype=float)
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) <= ROOT_IMAG_TOL].real
    in_range = real_roots[(real_roots >= -range_tol) & (real_roots <= 1.0 + range_tol)]
    residuals = np.abs(np.polyval(coeffs, in_range))
    valid = in_range[residuals <= residual_tol]
    if len(valid) == 0:
        return float("nan")
    return float(np.clip(np.max(valid), 0.0, 1.0))


def scale_label(scale_info):
    """Return a compact label for the max-L normalization."""
    return f"max_L scale={float(scale_info['scale_down']):g}"


def solve_info_label(info):
    """Return compact warning-aware solve metadata for notebook diagnostics."""
    if info is None:
        return "no solve info"
    warn = info.get("warn_patterns") or []
    return f"raw_ok={info.get('ok_raw')} clean={info.get('ok_clean')} warnings={warn}"


def warning_aware_solve(solving_fun, *args, warning_patterns=None, **kwargs):
    """Mark feasible solves with matching warning patterns as non-clean for boundary-sensitive checks."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = solving_fun(*args, **kwargs)
    warning_messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    warning_text = "\n".join([
        stdout_buffer.getvalue(),
        stderr_buffer.getvalue(),
        *warning_messages,
    ])
    active_patterns = SDPA_WARNING_PATTERNS if warning_patterns is None else list(
        warning_patterns
    )
    hits = [pattern for pattern in active_patterns if pattern in warning_text]
    ok_raw = bool(result[0]) if result is not None else False
    return {
        "result": result,
        "ok_raw": ok_raw,
        "ok_clean": ok_raw and not hits,
        "warn_patterns": hits,
    }


def warning_aware_bisection(l, r, tol, solving_fun, *args, warning_patterns=None, **kwargs):
    """Run bisection using only midpoint solves with no matching warning patterns."""
    best_x = None
    best_result = None
    warned_midpoints = 0
    while r - l > tol:
        m = 0.5 * (l + r)
        solve_info = warning_aware_solve(
            solving_fun,
            m,
            *args,
            warning_patterns=warning_patterns,
            **kwargs,
        )
        if solve_info["warn_patterns"]:
            warned_midpoints += 1
        if solve_info["ok_clean"]:
            r = m
            best_x = m
            best_result = solve_info["result"]
        else:
            l = m

    if best_x is None or best_result is None:
        return None, None, None, warned_midpoints
    return best_x, best_result[1], best_result[2], warned_midpoints


def worker_L_kappa_configs(L_values, kappa_values, n_workers, *, dedup_permutations=True):
    """Return per-worker (L, kappa) tuples, deduplicating worker permutations by default."""
    from itertools import product, combinations_with_replacement

    if n_workers <= 0:
        raise ValueError("n_workers must be positive.")

    worker_params = list(product(L_values, kappa_values))
    if dedup_permutations:
        worker_configs = combinations_with_replacement(worker_params, n_workers)
    else:
        worker_configs = product(worker_params, repeat=n_workers)

    out = []
    for w_cfg in worker_configs:
        L_cfg = tuple(p[0] for p in w_cfg)
        kappa_cfg = tuple(p[1] for p in w_cfg)
        out.append((L_cfg, kappa_cfg))
    return out
