import numpy as np
from PEPit import PEP, Point
from PEPit.constraint import Constraint
from PEPit.functions import SmoothStronglyConvexFunction, SmoothStronglyConvexQuadraticFunction

from utils import compress
from theory_helpers import optimal_lyapunov


def worst_case_performance(mu, L, eta, delta, method, gamma=None, n_workers=1, n_iterations=1, P=None, p=None, use_simplified=False, homogenous=False, zero_coefs=None, linear_compressor=False, verbose=0):
    """Solve a PEPit worst-case bound for EF, EF21, or EControl.

    Args:
        mu: Sequence of per-worker strong convexity parameters.
        L: Sequence of per-worker smoothness parameters.
        eta: EF/EF21 step size; for EControl, coefficient on the error state
            in the compressed control term.
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('EF', 'EF21', or 'EControl').
        gamma: EControl step size used in the x-update (required when method='EControl').
        n_workers: Number of workers.
        n_iterations: Number of iterations to perform.
        P: Lyapunov matrix used with `p` to define the performance metric.
        p: Per-worker function-value Lyapunov coefficients; required when `P` is provided.
        use_simplified: Whether to use a method-specific simplified Lyapunov metric.
        homogenous: If True, constrain all workers to share the same stationary point.
        zero_coefs: Optional list of (i, j) coefficient indices to zero out in P before evaluation.
        linear_compressor: If True, add sum-contraction constraints for a shared linear compressor.
        verbose: Verbosity passed to PEPit and exception logging.

    Returns:
        Worst-case performance value from PEPit, `np.nan` when P contains NaNs,
        or `None` if the PEP solve raises.

    Raises:
        ValueError: If `method` is unsupported or if `gamma` is missing for EControl.
    """
    if P is not None and np.isnan(P).any():
        return np.nan
    if P is not None and zero_coefs is not None:
        P = np.array(P, copy=True)
        for i, j in zero_coefs:
            P[i, j] = 0
            P[j, i] = 0

    # Setup PEP
    problem = PEP()
    funcs = []
    xs = []
    fs = []
    g0s = []
    x0 = problem.set_initial_point(name='x0')
    for i in range(n_workers):
        funcs.append(problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=mu[i], L=L[i]))
        xs.append(funcs[i].stationary_point(name=f'xs{i}'))
        fs.append(funcs[i](xs[i]))
        g0s.append(funcs[i].gradient(x0, name=f'g0{i}'))

    if homogenous:
        xref = xs[0]
        for xsi in xs[1:]:
            problem.set_initial_condition((xsi - xref)**2 == 0)

    def compress_workers(contents, name_prefix):
        cs = [compress(contents[i], problem, delta, name=f'{name_prefix}, worker {i}') for i in range(n_workers)]
        if linear_compressor and n_workers > 1:
            sum_contents = sum(contents, start=0*Point())
            sum_cs = sum(cs, start=0*Point())
            constraint = (sum_contents - sum_cs)**2 <= (1 - delta) * sum_contents**2
            problem.add_constraint(constraint, name=f'constraint: sum {name_prefix}')
        return cs

    # Run optimization method
    x = x0
    gs = g0s
    for j in range(n_iterations):
        if method == 'EF':
            if j == 0:
                es = [Point(name=f'e{j}, worker {i}') for i in range(n_workers)]
                cs = compress_workers([eta * gs[i] + es[i] for i in range(n_workers)], f'c{j}')
                c0s, e0s = cs.copy(), es.copy()
            else:
                cs = compress_workers([eta * gs[i] + es[i] for i in range(n_workers)], f'c{j}')
            es = [eta * gs[i] + es[i] - cs[i] for i in range(n_workers)]
            x = x - 1 / n_workers * sum(cs, start=0*Point())
            x.set_name('x1')
            gs = [funcs[i].gradient(x, name=f'g{j+1}, worker {i}') for i in range(n_workers)]
        elif method == 'EF21':
            if j == 0:
                cs = [Point(name=f'c{j}, worker {i}') for i in range(n_workers)]
                c0s = cs.copy()
            x = x - eta / n_workers * sum(cs, start=0*Point())
            x.set_name(f'x{j+1}')
            gs = [funcs[i].gradient(x, name=f'g{j+1}, worker {i}') for i in range(n_workers)]
            cos = compress_workers([gs[i] - cs[i] for i in range(n_workers)], f'compressed term {j+1}')
            cs = [cs[i] + cos[i] for i in range(n_workers)]
        elif method == 'EControl':
            if gamma is None:
                raise ValueError("gamma must be provided for method='EControl'")
            if j == 0:
                es = [Point(name=f'e{j}, worker {i}') for i in range(n_workers)]
                hs = [Point(name=f'h{j}, worker {i}') for i in range(n_workers)]
                cs = compress_workers([eta * es[i] + gs[i] - hs[i] for i in range(n_workers)], f'c{j}')
                c0s, e0s, h0s = cs.copy(), es.copy(), hs.copy()
            else:
                cs = compress_workers([eta * es[i] + gs[i] - hs[i] for i in range(n_workers)], f'c{j}')
            hs = [hs[i] + cs[i] for i in range(n_workers)]
            x = x - gamma / n_workers * sum(hs, start=0*Point())
            x.set_name(f'x{j+1}')
            es = [es[i] + gs[i] - hs[i] for i in range(n_workers)]
            gs = [funcs[i].gradient(x, name=f'g{j+1}, worker {i}') for i in range(n_workers)]
        else:
            raise ValueError(f'Method "{method}" not supported. Supported methods: EF, EF21, EControl')

    # Compute function values
    f0s = [funcs[i](x0) for i in range(n_workers)]
    fis = [funcs[i](x) for i in range(n_workers)]

    # Set up performance metric based on available information
    if P is not None:
        if method == "EF":
            diffs = [x0 - xsi for xsi in xs]
            for i, po in enumerate(diffs):
                po.set_name(f'x{i} - x*')
            init_parameters = np.array(diffs + g0s + c0s + e0s)
            cs = compress_workers([eta * gs[i] + es[i] for i in range(n_workers)], f'c{n_iterations}')
            parameters = np.array([x - xsi for xsi in xs] + gs + cs + es)
        elif method == "EF21":
            init_parameters = np.array([x0 - xsi for xsi in xs] + g0s + c0s)
            parameters = np.array([x - xsi for xsi in xs] + gs + cs)
        elif method == "EControl":
            diffs = [x0 - xsi for xsi in xs]
            for i, po in enumerate(diffs):
                po.set_name(f'x{i} - x*')
            init_parameters = np.array(diffs + g0s + c0s + e0s + h0s)
            cs_next = compress_workers([eta * es[i] + gs[i] - hs[i] for i in range(n_workers)], f'c{n_iterations}')
            parameters = np.array([x - xsi for xsi in xs] + gs + cs_next + es + hs)
        
        initial_condition = lya_quad(P, p, init_parameters, f0s, fs)
        problem.set_initial_condition(initial_condition <= 1, name='initial_condition')
        performance_metric = lya_quad(P, p, parameters, fis, fs)
        problem.set_performance_metric(performance_metric)
    elif use_simplified and method == 'EF':
        # V = sum_i ||x - xs - ei||^2 + 1/(n*sqrt(1-delta)) * ||sum_i ei||^2
        x_idx = 1 if n_workers > 1 else 0
        xs_ref = xs[x_idx]
        coef_cross = 1 / (n_workers * np.sqrt(1 - delta))
        coef_diag = 1 + coef_cross

        def simplified_lyapunov_ef(x, e_list):
            x_diff = x - xs_ref
            sum_e = sum(e_list, start=0*Point())
            term1 = n_workers * x_diff**2
            term2 = -2 * x_diff * sum_e
            term3 = coef_diag * sum([e**2 for e in e_list])
            term4 = 0
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    term4 += 2 * coef_cross * (e_list[i] * e_list[j])
            return term1 + term2 + term3 + term4

        init_cond = simplified_lyapunov_ef(x0, e0s)
        problem.set_initial_condition(init_cond <= 1, name='initial_condition')
        performance_metric = simplified_lyapunov_ef(x, es)
        problem.set_performance_metric(performance_metric)

    elif use_simplified and method == 'EF21' and linear_compressor:
        eps = 1 - delta
        sum_g0 = sum(g0s, start=0*Point())
        sum_c0 = sum(c0s, start=0*Point())
        sum_g = sum(gs, start=0*Point())
        sum_c = sum(cs, start=0*Point())
        init_cond = (sum_g0 - sum_c0)**2 + np.sqrt(eps) * sum_g0**2
        problem.set_initial_condition(init_cond <= 1, name='initial_condition')
        performance_metric = (sum_g - sum_c)**2 + np.sqrt(eps) * sum_g**2
        problem.set_performance_metric(performance_metric)

    elif use_simplified and method == 'EF21':
        if n_workers == 2:
            s1 = mu[0] + L[0]
            s2 = mu[1] + L[1]
            s = s1 + s2
            
            init_cond = (
                1 / s1 * (g0s[0] - c0s[0])**2
                + 1 / s2 * (g0s[1] - c0s[1])**2
                + np.sqrt(1 - delta) / s * (g0s[0] + g0s[1])**2
            )
            problem.set_initial_condition(init_cond <= 1, name='initial_condition')
            performance_metric = (
                1 / s1 * (gs[0] - cs[0])**2
                + 1 / s2 * (gs[1] - cs[1])**2
                + np.sqrt(1 - delta) / s * (gs[0] + gs[1])**2
            )
            problem.set_performance_metric(performance_metric)
            
        else:
            s_list = [mu[i] + L[i] for i in range(n_workers)]
            s = sum(s_list)
            scale = s / n_workers
            init_cond = scale * (
                sum([(1 / s_list[i]) * (g0s[i] - c0s[i])**2 for i in range(n_workers)])
                + np.sqrt(1 - delta) / s * (sum(g0s, start=0*Point()))**2
            )
            problem.set_initial_condition(init_cond <= 1, name='initial_condition')
            performance_metric = scale * (
                sum([(1 / s_list[i]) * (gs[i] - cs[i])**2 for i in range(n_workers)])
                + np.sqrt(1 - delta) / s * (sum(gs, start=0*Point()))**2
            )
            problem.set_performance_metric(performance_metric)
        
    try:
        rho = problem.solve(
            return_primal_or_dual="primal",
            solver="MOSEK",
            verbose=verbose,
        )
    except Exception as e:
        if verbose > 0:
            print(e)
        return None

    return rho


def lya_quad(P, p, params, fx, fs):
    """Build the Lyapunov expression from quadratic and function-value terms.

    Args:
        P: Lyapunov matrix.
        p: Per-worker coefficients for function-value gaps.
        params: Array of PEPit Points used in the quadratic term.
        fx: Current per-worker function values.
        fs: Per-worker stationary-point function values.

    Returns:
        PEPit expression for the Lyapunov value.
    """
    lya = 0
    for i, pi in enumerate(p):
        lya += pi * (fx[i] - fs[i])
    for i in range(len(P)):
        for j in range(len(P)):
            if isinstance(params[i], Point) and isinstance(params[j], Point):
                lya += P[i, j] * params[i] * params[j]
    
    return lya
