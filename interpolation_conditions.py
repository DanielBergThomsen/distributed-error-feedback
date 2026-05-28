'''
This file contains interpolation conditions for different function classes. It also contains
methods to generate conditions for a set of points automatically.

The code originates from https://github.com/bgoujaud/cycles/blob/master/tools/interpolation_conditions.py
'''

from math import inf
import cvxpy as cp


def inner_product(u, v):
    """Return the symmetric Gram-matrix coefficient for <u, v>."""
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    """Return the Gram-matrix coefficient representing ||u||^2."""
    return inner_product(u, u)


def smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L):
    """Computes interpolation conditions for smooth strongly convex functions between two points.

    Args:
        pointi: Tuple containing (xi, gi, fi) for the first point.
        pointj: Tuple containing (xj, gj, fj) for the second point.
        mu: Strong convexity parameter.
        L: Smoothness parameter.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Function-value coefficient fj - fi, scalar or vector matching fi/fj

    Raises:
        ValueError: If L or mu is None.
    """
    if L is None or mu is None:
        raise ValueError("L and mu must be provided for smooth strongly convex interpolation")

    xi, gi, fi = pointi
    xj, gj, fj = pointj

    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
    F = fj - fi

    return G, F


def lipschitz_operator_interpolation_i_j(pointi, pointj, L):
    """Computes interpolation conditions for Lipschitz operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        L: Lipschitz constant.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If L is None.
    """
    if L is None:
        raise ValueError("L must be provided for lipschitz operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L ** 2 * square(xi - xj)
    F = 0

    return G, F


def strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu):
    """Computes interpolation conditions for strongly monotone operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        mu: Strong monotonicity parameter.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If mu is None.
    """
    if mu is None:
        raise ValueError("mu must be provided for strongly monotone operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = mu * square(xi - xj) - inner_product(gi - gj, xi - xj)
    F = 0

    return G, F


def cocoercive_operator_interpolation_i_j(pointi, pointj, L):
    """Computes interpolation conditions for cocoercive operators between two points.

    Args:
        pointi: Tuple containing (xi, gi, _) for the first point.
        pointj: Tuple containing (xj, gj, _) for the second point.
        L: Inverse cocoercivity constant used in ||gi - gj||^2 <= L <xi - xj, gi - gj>.

    Returns:
        Tuple containing:
            - G: Matrix representing the interpolation condition
            - F: Always returns 0 for this operator type

    Raises:
        ValueError: If L is None.
    """
    if L is None:
        raise ValueError("L must be provided for cocoercive operator interpolation")

    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L * inner_product(xi - xj, gi - gj)
    F = 0

    return G, F


def interpolation(list_of_points, function_class, mu=None, L=None):
    """Generates interpolation conditions for a set of points based on the function class.

    Args:
        list_of_points: List of tuples, each containing (x, g, f) for a point.
        function_class: String specifying the function class. Must be one of:
            - "smooth strongly convex"
            - "lipschitz strongly monotone operator"
            - "strongly monotone operator"
            - "cocoercive operator"
        mu: Strong convexity/monotonicity parameter.
        L: Smoothness/Lipschitz parameter.

    Returns:
        Tuple containing:
            - list_of_matrices: List of matrices representing interpolation conditions
            - list_of_vectors: List of vectors representing function value differences
    """
    list_of_matrices = []
    list_of_vectors = []

    for i, pointi in enumerate(list_of_points):
        for j, pointj in enumerate(list_of_points):
            if i != j:
                if function_class == "smooth strongly convex":
                    G, F = smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)
                elif function_class == "lipschitz strongly monotone operator":
                    G, F = lipschitz_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "strongly monotone operator":
                    assert L == inf
                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "cocoercive operator":
                    assert mu == 0
                    G, F = cocoercive_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

    return list_of_matrices, list_of_vectors


def interpolation_combination(list_of_points, function_class, mu=None, L=None):
    """Form a linear combination of interpolation inequalities with CVXPY multipliers.

    Args:
        list_of_points: List of tuples, each containing (x, g, f) for a point.
        function_class: String specifying the function class.
        mu: Strong convexity/monotonicity parameter.
        L: Smoothness/Lipschitz parameter.

    Returns:
        Tuple containing:
            - matrix_combination: Linear combination of interpolation matrices
            - vector_combination: Linear combination of interpolation vectors
            - dual: Unconstrained CVXPY multiplier vector
    """
    list_of_matrices, list_of_vectors = interpolation(list_of_points, function_class, mu, L)
    nb_constraints = len(list_of_matrices)
    dual = cp.Variable((nb_constraints,))
    matrix_combination = cp.sum([dual[i] * list_of_matrices[i] for i in range(nb_constraints)])
    vector_combination = cp.sum([dual[i] * list_of_vectors[i] for i in range(nb_constraints)])

    return matrix_combination, vector_combination, dual


def compression_interpolation(compression_point, approximated_object, delta, non_expansive=False):
    """Return Gram-matrix constraints for a contractive compressor.

    The base condition is ||x - c||^2 <= (1 - delta) ||x||^2, where
    `approximated_object` is x and `compression_point` is c.

    Args:
        compression_point: Compressed point c = C(x).
        approximated_object: Object x being compressed.
        delta: Compression parameter between 0 and 1.
        non_expansive: If True, also return ||c||^2 <= ||x||^2.

    Returns:
        The base compression matrix, or `(base_matrix, non_expansive_matrix)` when
        `non_expansive=True`.

    Raises:
        ValueError: If delta is not between 0 and 1.
    """
    if delta < 0 or delta > 1:
        raise ValueError("delta must be between 0 and 1")

    c1 = square(approximated_object - compression_point) - (1 - delta) * square(approximated_object)
    if non_expansive:
        c2 = square(compression_point) - square(approximated_object)
        return c1, c2
    else:
        return c1
