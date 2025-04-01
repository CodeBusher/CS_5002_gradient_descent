import gradient_descent_one_variable as gd
import plot as p
import numpy as np
import contextlib


def f1(x):
    """f(x) = x * x"""
    x = np.asarray(x)
    return x * x


def deriv_f1(f, x):
    """f(x) = x * x, f'(x) = 2 * x"""
    x = np.asarray(x)
    return 2 * x


def f2(x):
    """f(x) = x * x - 2 * x + 3"""
    x = np.asarray(x)
    return x * x - 2 * x + 3


def deriv_f2(f, x):
    """f(x) = x * x - 2 * x + 3, f'(x) = 2 * x - 2"""
    x = np.asarray(x)
    return 2 * x - 2


def test_gradient_descent_one_variable():
    print('Testing gradient descent algorithm for one variable:')

    x0, alpha, eps = 3, 0.1, 0.001

    f1_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f1, f1)
    p.plot_opt(f1, f1_opt_x, 'Figure_1_f1')

    f2_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f2, f2)
    p.plot_opt(f2, f2_opt_x, 'Figure_2_f2')

    print('\n')


def test_gradient_descent_one_variable_deriv_a():
    print('Testing gradient descent algorithm by derivative approximation:')

    x0, alpha, eps = 3, 0.1, 0.001

    f1_opt_x_e = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f1, f1)
    f1_opt_x_a = gd.gradient_descent_one_variable(x0, alpha, eps, gd.derivative_approximation_one_variable, f1)
    print('-'*20)
    print(f1.__doc__)
    print('f1_opt_x_e =', f1_opt_x_e)
    print('f1_opt_x_a =', f1_opt_x_a)
    print('-' * 20)

    f2_opt_x_e = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f2, f2)
    f2_opt_x_a = gd.gradient_descent_one_variable(x0, alpha, eps, gd.derivative_approximation_one_variable, f2)
    print('-'*20)
    print(f2.__doc__)
    print('f2_opt_x_e =', f2_opt_x_e)
    print('f2_opt_x_a =', f2_opt_x_a)
    print('-' * 20)


def main():
    test_gradient_descent_one_variable()
    test_gradient_descent_one_variable_deriv_a()


if __name__ == '__main__':
    with open('gradient_descent_one_variable.txt', 'w') as file:
        with contextlib.redirect_stdout(file):
            main()