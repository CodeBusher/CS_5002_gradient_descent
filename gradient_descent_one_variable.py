def gradient_descent_one_variable(x0, alpha, eps, deriv_f, f, iter_max=1000):
    """
    Gradient descent is an iterative optimization algorithm used to estimate the minimum of a function.
    The main idea is to take repeated steps in the direction of the steepest decrease of the function.
    x_k+1 = x_k - alpha * f'(x_k) update until |x_k+1 - x_k| < epsilon
    :param x0: initial point x0
    :param alpha: the learning rate, the value of a step in the direction of the derivative
    :param eps: epsilon, the value of tolerance
    :param deriv_f: derivative function
    :param f: the one-variable function
    :param iter_max: the maximum of iteration numbers, avoiding infinite loops
    :return: the optimal x, where f(x) is the local minimum of the function
    """

    x_new = x0

    # x_k+1 = x_k - alpha * f'(x_k), update until |x_k+1 - x_k| < epsilon or the maximum number of iterations is reached
    for iter in range(iter_max):
        x_current = x_new
        x_new = x_current - alpha * deriv_f(f, x_current)
        # print(f'iter = {iter}, x = {x_current}')
        if abs(x_new - x_current) < eps:
            break

    print(f'iter_total = {iter+1}, x_optimal = {x_new}')

    # Return the optimal x where f(x) is the local minimum
    return x_new


def derivative_approximation_one_variable(f, x, h=1e-6):
    """f'(x) = [f(x+h) - f(x)] / h"""
    return (f(x+h) - f(x)) / h