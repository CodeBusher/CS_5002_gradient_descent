import numpy as np


def derivative_approximation_multi_variable(f, x, index, h=1e-6):
    """
    Calculate the derivative of a multivariable function f at point x with respect to the variable at the given index.
    :param f: function, a multivariable function
    :param x: float/int or list, the point at which to evaluate the derivative
    :param index: int, the index of the variable with respect to which the derivative is computed
    :param h: float, the small increment for the finite difference
    :return: the approximate partial derivative of f at x with respect to the variable at the given index
    """
    x = np.asarray(x, dtype=float)
    x_h = np.copy(x)
    x_h[index] += h

    return (f(x_h) - f(x)) / h


def gradient_descent_multi_variable(x0, alpha, eps, deriv_f, f, iter_max=1000):
    """
    Perform gradient descent to minimize the multivariable function f.
    :param x0: initial point
    :param alpha: the learning rate, the value of a step in the direction of the derivative
    :param eps: epsilon, the value of tolerance
    :param deriv_f: derivative function
    :param f: the multi-variable function
    :param iter_max: the maximum of iteration numbers, avoiding infinite loops
    :return: the optimal x, where f(x) is the local minimum of the function
    """

    x_new = np.array(x0, dtype=float)

    for iter in range(iter_max):
        x_current = x_new.copy()

        # Update each parameter using its partial derivative.
        d = 0
        for i in range(len(x_current)):
            x_new[i] = x_current[i] - alpha * deriv_f(f, x_current, i)
            d = max(d, abs(x_new[i] - x_current[i]))

        # Check for convergence
        if d < eps:
            return x_new

    # Reach maximum iterations.
    return x_new

