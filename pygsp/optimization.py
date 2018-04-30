# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.optimization` module provides tools to solve convex
optimization problems on graphs.
"""

from pygsp import utils
from pygsp.filters import Filter


logger = utils.build_logger(__name__)


def _import_pyunlocbox():
    try:
        from pyunlocbox import functions, solvers
    except Exception:
        raise ImportError('Cannot import pyunlocbox, which is needed to solve '
                          'this optimization problem. Try to install it with '
                          'pip (or conda) install pyunlocbox.')
    return functions, solvers


def prox_tv(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, maxit=200, use_matrix=True, verbose = True):
    r"""
    Total Variation proximal operator for graphs.

    This function computes the TV proximal operator for graphs. The TV norm
    is the one norm of the gradient. The gradient is defined in the
    function :meth:`pygsp.graphs.Graph.grad`.
    This function requires the PyUNLocBoX to be executed.

    This function solves:

    :math:`sol = \min_{z} \frac{1}{2} \|x - z\|_2^2 + \gamma  \|x\|_{TV}`

    Parameters
    ----------
    x: int
        Input signal
    gamma: ndarray
        Regularization parameter
    G: graph object
        Graphs structure
    A: lambda function
        Forward operator, this parameter allows to solve the following problem:
        :math:`sol = \min_{z} \frac{1}{2} \|x - z\|_2^2 + \gamma  \| A x\|_{TV}`
        (default = Id)
    At: lambda function
        Adjoint operator. (default = Id)
    nu: float
        Bound on the norm of the operator (default = 1)
    tol: float
        Stops criterion for the loop. The algorithm will stop if :
        :math:`\frac{n(t) - n(t - 1)} {n(t)} < tol`
        where :math:`n(t) = f(x) + 0.5 \|x-y\|_2^2` is the objective function at iteration :math:`t`
        (default = :math:`10e-4`)
    maxit: int
        Maximum iteration. (default = 200)
    use_matrix: bool
        If a matrix should be used. (default = True)

    Returns
    -------
    sol: solution

    Examples
    --------

    """
    if A is None:
        def A(x):
            return x
    if At is None:
        def At(x):
            return x

    tight = 0
    l1_nu = 2 * G.lmax * nu

    if use_matrix:
        def l1_a(x):
            return G.Diff * A(x)

        def l1_at(x):
            return G.Diff * At(D.T * x)
    else:
        def l1_a(x):
            return G.grad(A(x))

        def l1_at(x):
            return G.div(x)

    functions, _ = _import_pyunlocbox()
    f = functions.norm_l1(A=l1_a, At=l1_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
    sol = f.prox(x, gamma)


def prox_tik(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, maxit=200, use_matrix=True):
    r"""
    Proximal tikhonov operator for graphs.
    
    This function computes the proximal tikhonov operator for graphs. The tikhonov norm
    is the 2 norm of the gradient. The gradient is defined in the
    function :meth:`pygsp.graphs.Graph.grad`.
    This function requires the PyUNLocBoX to be executed.

    This function solves:

        (1) :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \gamma \| \nabla x\|_2^2`

    Note the nice following relationship:

        (2) :math:`x^T L x = \| \nabla x\|_2^2`

    Also note that the solution to (1) may be phrased as the lowpass filter:

        (3) :math:`sol = h(L)x` with :math:`h(\lambda) := \frac{1}{1+\gamma\lambda}`

    In the case that the Fourier basis of L is available, (3) should be considered.

    Parameters
    ----------
    x: int
        Input signal
    gamma: ndarray
        Regularization parameter
    G: graph object
        Graphs structure
    A: lambda function
        Forward operator, this parameter allows to solve the following problem:
    :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \gamma \| \nabla A x\|_2^2`
        (default = Id)
    At: lambda function
        Adjoint operator. (default = Id)
    nu: float
        Bound on the norm of the operator (default = 1)
    tol: float
        Stops criterion for the loop. The algorithm will stop if :
        :math:`\frac{n(t) - n(t - 1)} {n(t)} < tol`
        where :math:`n(t) = f(x) + 0.5 \|x-y\|_2^2` is the objective function at iteration :math:`t`
        (default = :math:`10e-4`)
    maxit: int
        Maximum iteration. (default = 200)
    use_matrix: bool
        If the gradient matrix should be used. (default = True)

    Returns
    -------
    sol: solution

    Examples
    --------

    """
    G.estimate_lmax()
    
    if A is None:
        fi = Filter(G, lambda x: 1/(1+2*gamma*x))
        if hasattr(G, '_U'):
            method = 'exact'
        else:
            method = 'chebyshev'
        sol = fi.filter(x, method)
    else:
        if At is None:
            raise ValueError("Please supply an adjoint operator to prox_tik.")

    tight = 0
    l2_nu = 2 * G.lmax * nu

    if use_matrix:
    
        def l2_a(x):
            return G.D * A(x)

        def l2_at(x):
            return At(G.D.T * x)
    else:
        def l2_a(x):
            return G.grad(A(x))

        def l2_at(x):
            return At(G.div(x))

    functions, _ = _import_pyunlocbox()
    functions.norm_l2(x, gamma, A=l2_a, At=l2_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
