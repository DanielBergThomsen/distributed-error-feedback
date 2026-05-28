"""Helper functions for theoretical analysis of optimization methods.

This module provides functions for computing optimal parameters and Lyapunov functions
for various optimization methods including Error Feedback (EF), EF21, and Compressed Gradient
Descent (CGD).
"""

import numpy as np


def optimal_lyapunov(delta, method):
    """Computes the optimal Lyapunov matrix and function-value weights for a given method.

    This function returns the optimal Lyapunov function parameters for different
    optimization methods. The Lyapunov function is used to prove convergence
    guarantees for the respective methods.

    Args:
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('EF', 'EF21', or 'CGD').

    Returns:
        tuple: (P, p) where:
            - P: Lyapunov matrix (None for CGD)
            - p: List of function-value weights

    Raises:
        ValueError: If an invalid method is specified.
    """
    if method == 'EF':
        P = np.array([[1, -1], [-1, 1 + 1 / np.sqrt(1 - delta)]])
        p = [0.0]
    elif method == 'EF21':
        P = np.array([[1 + np.sqrt(1 - delta), -1], [-1, 1]])
        p = [0.0]
    elif method == 'CGD':
        P = None
        p = [1.0]
    else:
        raise ValueError(f"Method '{method}' not supported")

    return P, p


def optimal_step_size(mu, L, delta, method):
    """Computes the optimal step size for a given optimization method.

    This function calculates the optimal step size for different optimization
    methods based on the strong convexity parameter (mu), smoothness parameter (L),
    and compression parameter (delta).

    Args:
        mu: Strong convexity parameter, or a list aligned with L for EF/EF21.
        L: Smoothness parameter, or a list aligned with mu for EF/EF21.
        delta: Compression parameter between 0 and 1.
        method: Optimization method to use ('EF', 'EF21', or 'CGD').

    Returns:
        float: Optimal step size for the specified method.

    Raises:
        ValueError: If an invalid method is specified.
    """
    if method in ['EF', 'EF21']:
        # Handle per-worker parameters.
        if isinstance(mu, list):
            const_sums = [mu[i] + L[i] for i in range(len(mu))]
        else:
            const_sums = [mu + L]
        return 2 / np.mean(const_sums) * (delta / (1 + np.sqrt(1 - delta))**2)
    elif method == 'CGD':
        return 2 / ((1+np.sqrt(1 - delta)) * mu + (1+np.sqrt(1 - delta)) * L)
    else:
        raise ValueError(f"Method '{method}' not supported")
