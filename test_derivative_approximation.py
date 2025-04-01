import gradient_descent_one_variable as gdu
import gradient_descent_mul_variable as gdm
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


def f_x_y(x):
    """f(x, y) = x * x + y * y"""
    return x[0] ** 2 + x[1] ** 2


def deriv_f_x_y(f, x, index):
    """f(x, y) = x * x + y * y"""
    if index == 0:
        return 2 * x[0]
    else:
        return 2 * x[1]


def test_derivative_approximation_one_variable(f, deriv_f, deriv_fa, eps=1e-5):
    print('Test Derivative Approximation: ', f.__doc__)
    x_list = np.linspace(-50, 50, 100).tolist()
    is_equal = 0
    for x_test in x_list:
        d_f = deriv_f(f, x_test)
        d_fa = deriv_fa(f, x_test)
        is_equal += abs(d_f - d_fa) < eps
        print(f'x = {x_test:.4f}, derivative = {d_f:.4f}, derivative approximation= {d_fa:.4f}')

    if is_equal == len(x_list):
        print(f'All the values of the approximate derivative and exact derivative are equal.')
    else:
        print(f'Some values of the approximate derivative and exact derivative are not equal.')

    print('\n')


def test_derivative_approximation_two_variable(f, deriv_f, deriv_fa, eps=1e-5):
    print('Test Derivative Approximation: ', f.__doc__)
    x_vals = np.linspace(-50, 50, 100)
    y_vals = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    points = np.c_[X.ravel(), Y.ravel()].tolist()

    d_f = [0, 0]
    d_fa = [0, 0]
    is_equal = 0
    for point in points:
        flag = True
        for i in range(len(point)):
            d_f[i] = deriv_f(f, point, i)
            d_fa[i] = deriv_fa(f, point, i)
            flag = flag and abs(d_f[i] - d_fa[i]) < eps
        is_equal += flag

        x_formatted = ", ".join([f"{xi:.4f}" for xi in point])
        df_formatted = ", ".join([f"{df:.4f}" for df in d_f])
        dfa_formatted = ", ".join([f"{dfa:.4f}" for dfa in d_fa])

        print(f"position = ({x_formatted}), derivative = ({df_formatted}), derivative approximation = ({dfa_formatted})")

    if is_equal == len(points):
        print(f'All the values of the approximate derivative and exact derivative are almost equal.')
    else:
        print(f'Some values of the approximate derivative and exact derivative are not almost equal.')

    print('\n')


def main():
    test_derivative_approximation_one_variable(f=f1, deriv_f=deriv_f1, deriv_fa=gdu.derivative_approximation_one_variable)
    test_derivative_approximation_one_variable(f=f2, deriv_f=deriv_f2, deriv_fa=gdu.derivative_approximation_one_variable)
    test_derivative_approximation_two_variable(f=f_x_y, deriv_f=deriv_f_x_y, deriv_fa=gdm.derivative_approximation_multi_variable)


if __name__ == '__main__':
    with open('test_derivative_approximation.txt', 'w') as file:
        with contextlib.redirect_stdout(file):
            main()