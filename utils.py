"""Utility functions for optimization and parallel computation.

This module provides helper functions for compression operations, parallel computation
using Dask, and array manipulation utilities.
"""

from PEPit import Point
from dask import compute, delayed
from tqdm.dask import TqdmCallback
import numpy as np


def compress(o, problem, delta, name=None, equality=False):
    """Compresses an object by adding a constraint to the PEP problem.

    This function implements the compression operator as described in the paper
    "Sparsified SGD with Memory" by Stich et al. It adds a constraint of the form
    (o - c)^2 <= (1 - delta)o^2 to the PEP problem, where c is the compressed object.

    References:
        Stich, S., Cordonnier, J-B., & Jaggi, M. (2018). 
        Sparsified SGD with Memory.
        Conference on Neural Information Processing Systems (NeurIPS 2018).
        https://papers.nips.cc/paper_files/paper/2018/hash/b440509a0106086a67bc2ea9df0a1dab-Abstract.html

    Args:
        o: The original object to compress.
        problem: The PEP problem to add the constraint to.
        delta: The compression parameter between 0 and 1.
        name: Optional name for the compressed point.
        equality: If True, uses equality constraint instead of inequality.

    Returns:
        Point: The compressed object.

    Raises:
        ValueError: If delta is not between 0 and 1.
    """
    if not 0 <= delta <= 1:
        raise ValueError("Compression parameter delta must be between 0 and 1")

    c = Point(name=name)

    if equality:
        constraint = (o - c)**2 == (1-delta)*o**2
    else:
        constraint = (o - c)**2 <= (1-delta)*o**2
    problem.add_constraint(constraint, name=f"constraint: {name}")

    return c


def dask_grid_compute(func, xs, ys, show_progress=True, scheduler="processes", num_workers=None):
    """Computes a function over a 2D grid of values in parallel using Dask.
    
    This function evaluates the given function at each point in the 2D grid defined
    by xs and ys, using parallel computation through Dask. It handles both scalar
    and array return values from the function. The first grid point is evaluated
    once up front to infer output shapes, so func should be side-effect-free.

    Args:
        func: Function to compute over the grid. Can return a single value or a tuple of values.
        xs: 2D array of x values.
        ys: 2D array of y values.
        show_progress: Whether to show a progress bar during computation.
        scheduler: Dask scheduler name.
        num_workers: Optional worker cap passed to Dask process/thread schedulers.
        
    Returns:
        If func returns a single value: numpy array of shape (xs.shape + func_return_shape)
        If func returns a tuple: tuple of numpy arrays, each of shape (xs.shape + corresponding_return_shape)

    Raises:
        ValueError: If xs and ys have different shapes.
    """
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")

    indices = [(i, j) for i in range(xs.shape[0]) for j in range(xs.shape[1])]

    # Get a sample output to determine shapes
    sample_output = func(xs[0, 0], ys[0, 0])
    
    # Determine output shapes based on sample
    if isinstance(sample_output, tuple):
        output_shapes = []
        for item in sample_output:
            if hasattr(item, 'shape'):
                output_shapes.append(xs.shape + item.shape)
            else:
                output_shapes.append(xs.shape)
    else:
        if hasattr(sample_output, 'shape'):
            output_shapes = [xs.shape + sample_output.shape]
        else:
            output_shapes = [xs.shape]

    # Create delayed objects for parallel computation
    delayed_results = [delayed(func)(xs[i, j], ys[i, j]) for i, j in indices]
    
    # Execute computations
    compute_kwargs = {"scheduler": scheduler}
    if num_workers is not None:
        compute_kwargs["num_workers"] = int(num_workers)

    if show_progress:
        with TqdmCallback(desc="compute"):
            computed_results = compute(*delayed_results, **compute_kwargs)
    else:
        computed_results = compute(*delayed_results, **compute_kwargs)

    # Process results
    if isinstance(computed_results[0], tuple):
        results = [np.zeros(shape) for shape in output_shapes]
        for (i, j), result_tuple in zip(indices, computed_results):
            for k, result in enumerate(result_tuple):
                if hasattr(result, 'shape'):
                    results[k][i, j] = result
                else:
                    results[k][i, j] = result
        return tuple(results)
    else:
        results = np.zeros(output_shapes[0])
        for (i, j), result in zip(indices, computed_results):
            if hasattr(result, 'shape'):
                results[i, j] = result
            else:
                results[i, j] = result
        return results


def dask_parallel_map(func, xs, scheduler="processes", show_progress=True, num_workers=None):
    """Maps a function over a sequence of values in parallel using Dask.
    
    Args:
        func: Function to apply to each element.
        xs: Sequence of values to process.
        scheduler: Dask scheduler name.
        show_progress: Whether to show a progress bar during computation.
        num_workers: Optional worker cap passed to Dask process/thread schedulers.
        
    Returns:
        tuple: Results of applying func to each element in xs, in input order.
    """
    delayed_results = [delayed(func)(x) for x in xs]
    compute_kwargs = {"scheduler": scheduler}
    if num_workers is not None:
        compute_kwargs["num_workers"] = int(num_workers)

    if show_progress:
        with TqdmCallback(desc="compute"):
            res = compute(*delayed_results, **compute_kwargs)
    else:
        res = compute(*delayed_results, **compute_kwargs)
    return res


def nan_greater_than(array, threshold):
    """Replaces values greater than threshold with NaN in a copy of the array.
    
    Args:
        array: Input numpy array.
        threshold: Threshold value. Values greater than this will be replaced with NaN.
        
    Returns:
        numpy.ndarray: Copy of input array with values > threshold replaced by NaN.
    """
    copy = array.copy()
    copy[copy > threshold] = np.nan
    return copy
