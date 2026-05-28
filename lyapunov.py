'''Lyapunov feasibility checks and bisection utilities for compressed optimization methods.

The implementation was based on the following repo by Baptiste Goujaud:
https://github.com/bgoujaud/cycles
'''

import cvxpy as cp
import numpy as np
from interpolation_conditions import interpolation_combination, compression_interpolation, square
import itertools


def require_cvxpy_solver(solver):
    """Raise before solve if a requested CVXPY solver is unavailable."""
    solver_name = str(solver).upper()
    available = set(cp.installed_solvers())
    if solver_name not in available:
        available_text = ", ".join(sorted(available)) or "<none>"
        raise RuntimeError(
            f"Required solver {solver_name!r} is not installed in this Python environment. "
            f"Available CVXPY solvers: {available_text}."
        )


def has_lyapunov(rho, eta, delta, n_workers=1,
                 function_class='smooth strongly convex', mus=None, Ls=None, 
                 method='EF', zero_coefs=None, use_residual=True, 
                 log_det_iterations=0, log_det_delta=1e-6, 
                 use_simplified_lyapunov=False,
                 homogenous=False,
                 gamma=None,
                 solver='MOSEK',
                 solve_kwargs=None,
                 ):
    """Checks if a Lyapunov function exists for a desired rate of convergence.

    Args:
        rho: Desired rate of convergence.
        eta: EF/EF21 step size; for EControl, coefficient on the error state
            in the compressed control term.
        delta: Compression parameter between 0 and 1.
        n_workers: Number of workers for distributed optimization.
        function_class: Type of function class to consider. Options:
            - 'smooth strongly convex'
            - 'lipschitz strongly monotone operator'
            - 'strongly monotone operator'
            - 'cocoercive operator'
        mus: Strong convexity/monotonicity parameters, one per worker.
        Ls: Smoothness/Lipschitz parameters, one per worker.
        method: Optimization method to use. Options: 'EF', 'EF21', 'EControl'.
        zero_coefs: List of tuples (i,j) specifying which coefficients to set to zero.
        use_residual: Whether to use residual terms in the Lyapunov function.
        log_det_iterations: Number of iterations for log determinant optimization.
        log_det_delta: Small constant for numerical stability in log determinant.
        use_simplified_lyapunov: Whether to use the implemented simplified
            Lyapunov structure for EF/EF21.
        homogenous: Whether to use the homogeneous-worker normalization.
        gamma: EControl step size used in the x-update.
        solver: CVXPY solver name passed to `problem.solve`.
        solve_kwargs: Extra keyword arguments passed to `problem.solve` (e.g. mosek_params).

    Returns:
        Tuple containing:
            - bool: Whether a valid Lyapunov function exists.
            - np.ndarray: Matrix P of the Lyapunov function if found, otherwise NaN.
            - np.ndarray | float: Vector p if found, otherwise NaN.
    """
    require_cvxpy_solver(solver)

    # Initialize basis vectors
    if method == 'EControl':
        d = 1 + n_workers * 7
        basis_vectors = np.eye(d)
        x0 = basis_vectors[0]
        e0, g0, c0, h0, g1, c1, xs = [
            basis_vectors[1 + i * n_workers:1 + (i + 1) * n_workers] for i in range(7)
        ]
    else:
        d = 1 + n_workers * 6
        basis_vectors = np.eye(d)
        x0 = basis_vectors[0]
        e0, g0, c0, g1, c1, xs = [
            basis_vectors[1 + i * n_workers:1 + (i + 1) * n_workers] for i in range(6)
        ]

    if n_workers == 1:
        xs = [np.zeros(d)]  # Set the single reference point to the origin WLOG.
    elif homogenous:
        # Homogeneous workers share the origin WLOG.
        xs = [np.zeros(d) for _ in range(n_workers)]
    gs = np.zeros(d)
    
    # Initialize function value basis vectors
    d_f = n_workers * 2
    f_basis_vectors = np.eye(d_f)
    f0, f1 = [f_basis_vectors[i * n_workers:(i + 1) * n_workers] for i in range(2)]
    fs = np.zeros(d_f)

    # Start maintaining list of supplementary constraints
    supplementary_constraints_decrease = []

    # Setup based on method
    if method == 'EF':
        x1 = x0 - 1 / n_workers * np.sum(c0, axis=0)
        e1 = [e0[i] + eta * g0[i] - c0[i] for i in range(n_workers)]
        supplementary_constraints_decrease.extend([compression_interpolation(c0[i], e0[i] + eta * g0[i], delta) for i in range(n_workers)])
        supplementary_constraints_decrease.extend([compression_interpolation(c1[i], e1[i] + eta * g1[i], delta) for i in range(n_workers)])
        P_dim = 4 * n_workers
        xi0 = np.array([x0 - xs[i] for i in range(n_workers)] + [g0[i] for i in range(n_workers)] + [c0[i] for i in range(n_workers)] + [e0[i] for i in range(n_workers)])
        xi1 = np.array([x1 - xs[i] for i in range(n_workers)] + [g1[i] for i in range(n_workers)] + [c1[i] for i in range(n_workers)] + [e1[i] for i in range(n_workers)])

    elif method == 'EF21':
        x1 = x0 - eta / n_workers * np.sum(e0, axis=0)
        e1 = [np.zeros(d) for _ in range(n_workers)]
        for i in range(n_workers):
            e1[i] = e0[i] + c0[i]
        supplementary_constraints_decrease.extend([compression_interpolation(c0[i], g1[i] - e0[i], delta) for i in range(n_workers)])
        P_dim = 3 * n_workers
        xi0 = np.array([x0 - xs[i] for i in range(n_workers)] + [g0[i] for i in range(n_workers)] + [e0[i] for i in range(n_workers)])
        xi1 = np.array([x1 - xs[i] for i in range(n_workers)] + [g1[i] for i in range(n_workers)] + [e1[i] for i in range(n_workers)])

    elif method == 'EControl':
        if gamma is None:
            raise ValueError("gamma must be provided for EControl.")
        x1 = x0 - gamma / n_workers * np.sum(h0 + c0, axis=0)
        e1 = [e0[i] + g0[i] - h0[i] - c0[i] for i in range(n_workers)]
        h1 = [h0[i] + c0[i] for i in range(n_workers)]
        supplementary_constraints_decrease.extend([
            compression_interpolation(c0[i], eta * e0[i] + g0[i] - h0[i], delta) for i in range(n_workers)
        ])
        supplementary_constraints_decrease.extend([
            compression_interpolation(c1[i], eta * e1[i] + g1[i] - h1[i], delta) for i in range(n_workers)
        ])
        P_dim = 5 * n_workers
        xi0 = np.array(
            [x0 - xs[i] for i in range(n_workers)] +
            [g0[i] for i in range(n_workers)] +
            [c0[i] for i in range(n_workers)] +
            [e0[i] for i in range(n_workers)] +
            [h0[i] for i in range(n_workers)]
        )
        xi1 = np.array(
            [x1 - xs[i] for i in range(n_workers)] +
            [g1[i] for i in range(n_workers)] +
            [c1[i] for i in range(n_workers)] +
            [e1[i] for i in range(n_workers)] +
            [h1[i] for i in range(n_workers)]
        )

    else:
        raise ValueError(f"Invalid method: {method}")

    # Setup Lyapunov variables and decrease terms
    P = cp.Variable((P_dim, P_dim), symmetric=True)
    p = cp.Variable((n_workers,))
    VP = xi0.T @ P @ xi0
    VP_plus = xi1.T @ P @ xi1
    Vp = f0.T @ p
    Vp_plus = f1.T @ p

    # Setup problem constraints
    constraints = [p >= 0, P >> 0]
    
    # Add interpolation conditions
    matrix_combinations = []
    vector_combinations = []
    
    # Interpolation conditions (decrease)
    for i in range(n_workers):
        points = [(xs[i], gs[i], fs[i]), (x0, g0[i], f0[i]), (x1, g1[i], f1[i])]
        matrix_comb, vector_comb, dual = interpolation_combination(points, function_class, mus[i], Ls[i])
        matrix_combinations.append(matrix_comb)
        vector_combinations.append(vector_comb)
        constraints.append(dual >= 0)

    # Lyapunov decrease constraints
    if supplementary_constraints_decrease:
        compression_dual = cp.Variable((len(supplementary_constraints_decrease),))
        constraints.append(compression_dual >= 0)
        supplementary_term = cp.sum([
            compression_dual[i] * matrix
            for i, matrix in enumerate(supplementary_constraints_decrease)
        ])
    else:
        supplementary_term = 0
    constraints.append(VP_plus - rho * VP << cp.sum(matrix_combinations) + supplementary_term)
    constraints.append(Vp_plus - rho * Vp <= cp.sum(vector_combinations))

    # Apply custom Lyapunov function constraints
    if not use_residual:
        constraints.extend([cp.sum(p) == 0])
        if not use_simplified_lyapunov:
            constraints.extend([cp.trace(P) == 1])
    else:
        if not use_simplified_lyapunov:
            constraints.append(cp.trace(P) + cp.sum(p) == 1)

    # Apply sparsity constraints if specified
    if zero_coefs is not None:
        for i, j in zero_coefs:
            constraints.extend([P[i, j] == 0, P[j, i] == 0])

    # For homogeneous workers, remove redundant reference-point directions
    if homogenous and not (method == 'EF' and use_simplified_lyapunov):
        constraints.extend([P[i, j] == 0 for j in range(P.shape[0]) for i in range(n_workers-1)])

    # Apply method-specific simplified Lyapunov constraints
    if method == 'EF' and use_simplified_lyapunov:
        x_idx = 1 if n_workers > 1 else 0
        e_indices = list(range(3 * n_workers, 4 * n_workers))
        ef_cross = 1 / (n_workers * np.sqrt(1 - delta))
        
        constraints.extend([P[x_idx, x_idx] == n_workers])
        for i in e_indices:
            constraints.extend([P[i, i] == 1 + ef_cross])
            constraints.extend([P[x_idx, i] == -1])
            constraints.extend([P[i, x_idx] == -1])
            for j in range(e_indices.index(i) + 1, len(e_indices)):
                idx_j = e_indices[j]
                constraints.extend([P[i, idx_j] == ef_cross])
                constraints.extend([P[idx_j, i] == ef_cross])
                
        constraints.extend([cp.sum(p) == 0])

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if i == x_idx and j == x_idx: continue
                if i in e_indices and j in e_indices: continue
                if (i == x_idx and j in e_indices) or (j == x_idx and i in e_indices): continue
                constraints.extend([P[i, j] == 0])

    elif method == 'EF21' and use_simplified_lyapunov:
        if n_workers == 2:
            constraints.extend([P[0, i] == 0 for i in range(n_workers)])
            constraints.extend([P[i, 0] == 0 for i in range(n_workers)])
            constraints.extend([P[1, i] == 0 for i in range(n_workers)])
            constraints.extend([P[i, 1] == 0 for i in range(n_workers)])
            constraints.extend([P[-2, -1] == 0, P[-1, -2] == 0, P[2, -1] == 0, P[-1, 2] == 0, P[-2, -3] == 0, P[-3, -2] == 0])
            mubar = sum(mus)
            Lbar = sum(Ls)
            constraints.extend([P[4, 4] * (mus[0] + Ls[0]) == P[5, 5] * (mus[1] + Ls[1])])
            inv_hypo = 2 * ((mus[1] + Ls[1]) * np.sqrt(1 - delta) / (mubar + Lbar) + (mus[1] + Ls[1]) / (Ls[0] + mus[0]) + 1)
            constraints.extend([
                P[2, 2] == 1/2 - P[5, 5],
                P[3, 3] == 1/2 - P[4, 4],
                P[2, 3] == P[2, 2] + P[3, 3] - 1 / 2,
                P[3, 2] == P[2, 2] + P[3, 3] - 1 / 2,
                P[4, 4] >= 0,
                P[5, 5] == 1 / inv_hypo,
                P[2, 4] == P[3, 3] - 1/ 2,
                P[4, 2] == P[3, 3] - 1/ 2,
                P[3, 5] == P[2, 2] - 1/ 2,
                P[5, 3] == P[2, 2] - 1/ 2
            ])
        elif n_workers > 2:
            # V = (sqrt(eps)/n) ||sum g||^2 + (1/n) sum_i w_i ||g_i - e_i||^2,
            # with w_i = S / (L_i + mu_i), S = sum_i (L_i + mu_i).
            eps = 1 - delta
            A = np.sqrt(eps) / n_workers
            s_list = [Ls[i] + mus[i] for i in range(n_workers)]
            S = float(np.sum(s_list))
            weights = [S / s_i if s_i != 0 else 0.0 for s_i in s_list]
            g_indices = list(range(n_workers, 2 * n_workers))
            e_indices = list(range(2 * n_workers, 3 * n_workers))
            
            # Zero out x terms
            for i in range(n_workers):
                constraints.extend([P[i, j] == 0 for j in range(P.shape[0])])
                constraints.extend([P[j, i] == 0 for j in range(P.shape[0])])
            
            # Fill g and e terms
            for i in range(n_workers):
                g_i = g_indices[i]
                e_i = e_indices[i]
                coef = weights[i] / n_workers
                
                # Diag terms
                # ||sum g||^2 contributes A to g_i^2
                # w_i/n * ||g_i - e_i||^2 contributes coef to g_i^2 and e_i^2
                constraints.extend([P[g_i, g_i] == A + coef])
                constraints.extend([P[e_i, e_i] == coef])
                
                # Cross g_i, e_i from w_i/n * ||g_i - e_i||^2 (-2 <g,e>)
                constraints.extend([P[g_i, e_i] == -coef])
                constraints.extend([P[e_i, g_i] == -coef])
                
                # Cross terms for sum g
                for j in range(i + 1, n_workers):
                    g_j = g_indices[j]
                    constraints.extend([P[g_i, g_j] == A])
                    constraints.extend([P[g_j, g_i] == A])
            
            # Zero other off-diagonal g/e terms between different workers
            for i in range(n_workers):
                for j in range(n_workers):
                    if i == j: continue
                    constraints.extend([P[g_indices[i], e_indices[j]] == 0])
                    constraints.extend([P[e_indices[j], g_indices[i]] == 0])
                    constraints.extend([P[e_indices[i], e_indices[j]] == 0])
                    constraints.extend([P[e_indices[j], e_indices[i]] == 0])

            constraints.extend([cp.sum(p) == 0])

    # Solve the optimization problem
    objective = cp.Minimize(0)
    logdet_iters = log_det_iterations
    prob = cp.Problem(objective, constraints)
    P_value = np.full_like(P, np.nan)
    p_value = np.nan

    try:
        current_solve_kwargs = dict(solve_kwargs or {})
        current_solve_kwargs.pop('solver', None)
        prob_solved = logdet_solve(
            P,
            prob,
            logdet_iters,
            log_det_delta,
            solver=solver,
            **current_solve_kwargs,
        )
        P_value = P.value if prob_solved.status == 'optimal' else P_value
        p_value = p.value if prob_solved.status == 'optimal' else p_value
    except cp.error.SolverError:
        return False, P_value, p_value

    return prob_solved.status == 'optimal', P_value, p_value


def logdet_solve(P, problem, log_det_iterations, log_det_delta, **kwargs):
    """Solve the CVXPY problem, optionally using an iterative log-det heuristic.

    Args:
        P: Symmetric Lyapunov matrix variable used by the log-det heuristic.
        problem: CVXPY problem to solve.
        log_det_iterations: Number of log-det refinement iterations; 0 runs a single solve.
        log_det_delta: Diagonal regularization added before inverting the current P estimate.
        **kwargs: Additional keyword arguments passed to `problem.solve`.

    Returns:
        CVXPY problem after the final solve, with `.status` and `.value` populated.
    """
    if log_det_iterations == 0:
        problem.solve(**kwargs)
        return problem
    Pk = np.eye(P.shape[0])
    for i in range(log_det_iterations):
        W = np.linalg.inv(Pk + log_det_delta * np.eye(Pk.shape[0]))
        logdet_problem = cp.Problem(cp.Minimize(cp.trace(W @ P)), problem.constraints)
        logdet_problem.solve(**kwargs)
        if logdet_problem.status == 'optimal':
            Pk = P.value if P.value is not None else Pk
        else:
            break
    logdet_problem.solve(**kwargs)
    return logdet_problem


def bisection(l, r, tol, solving_fun, *args, **kwargs):
    """Perform bisection to find the smallest feasible rate within tolerance.

    Args:
        l: Left bound of the search interval.
        r: Right bound of the search interval.
        tol: Tolerance for the search.
        solving_fun: Function to solve at each iteration.
        *args: Additional positional arguments for solving_fun.
        **kwargs: Additional keyword arguments for solving_fun.

    Returns:
        Tuple containing:
            - float | None: Smallest feasible rate found within tolerance, or None.
            - np.ndarray | None: Matrix P returned by `solving_fun`, or None.
            - object | None: Third value returned by `solving_fun`, or None.
    """
    r_init = r
    while r - l > tol:
        m = (l + r) / 2
        feasible, P, pf = solving_fun(m, *args, **kwargs)
        if feasible:
            r = m
        else:
            l = m

    if r == r_init and not feasible:
        return None, None, None
    _, P, pf = solving_fun(r, *args, **kwargs)
    return r, P, pf


if __name__ == "__main__":
    mus = [0.1, 0.1]
    Ls = [1.0, 1.0]
    from theory_helpers import optimal_step_size
    delta = 0.50
    method = 'EF'
    eta = optimal_step_size(mus[0], Ls[0], delta=delta, method=method) * 0.3
    n_workers = 2
    rho, P1, p = bisection(0.0, 1.2, 1e-6, has_lyapunov, eta=eta, delta=delta, n_workers=n_workers, mus=mus, Ls=Ls, 
                          method = method, use_residual=False,
                          homogenous=True)
    print(rho)
    rho, P2, p = bisection(0.0, 1.2, 1e-6, has_lyapunov, eta=eta, delta=delta, n_workers=n_workers, mus=mus, Ls=Ls, 
                          method = method, use_residual=False,
                          homogenous=True, use_simplified_lyapunov=True)
    print(rho)
    if P1 is not None and P2 is not None:
        diff = P1 / np.trace(P1) - P2 / np.trace(P2)
        diff[np.abs(diff) < 1e-05] = 0
        np.set_printoptions(linewidth=1000, suppress=True)
        print(diff)
    else:
        print("Optimization failed (P1 or P2 is None)")
