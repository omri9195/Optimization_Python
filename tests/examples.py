import numpy as np

# Given that the functions are pre-defined:
# In the functions below we will store and if needed return a hardcoded hessian and gradient for computational purposes


def quadratic_function_1(x, compute_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def quadratic_function_2(x, compute_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def quadratic_function_3(x, compute_hessian=False):
    R = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    D = np.array([[100, 0], [0, 1]])
    Q = np.dot(np.dot(R.T, D), R)
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def rosenbrock_function(x, compute_hessian=False):
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])
    if compute_hessian:
        h = np.array([
            [-400 * (x[1] - 3 * x[0]**2) + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    else:
        h = None
    return f, g, h


def linear_function(x, compute_hessian=False):
    a = np.array([1.5, 2])  # My choice of a
    f = np.dot(a.T, x)
    g = a
    h = np.zeros((len(x), len(x))) if compute_hessian else None
    return f, g, h


def boyds_book_function(x, compute_hessian=False):
    f = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    ])
    if compute_hessian:
        h = np.array([
            [np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1),
             3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)],
            [3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1),
             9*np.exp(x[0] + 3*x[1] - 0.1) + 9*np.exp(x[0] - 3*x[1] - 0.1)]
        ])
    else:
        h = None
    return f, g, h

