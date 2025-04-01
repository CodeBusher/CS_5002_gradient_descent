import gradient_descent_one_variable as gd
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


def parameter_hyper_tuning_x0(x0_list, alpha, eps):
    print("Parameter Hyper Tuning x0:")

    print(f1.__doc__)
    for x0 in x0_list:
        f1_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f1, f1)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f1_opt_x}')
        print('-' * 20)

    print(f2.__doc__)
    for x0 in x0_list:
        f2_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f2, f2)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f2_opt_x}')
        print('-' * 20)
    print('\n')


def parameter_hyper_tuning_alpha(x0, alpha_list, eps):
    print("Parameter Hyper Tuning alpha:")

    print(f1.__doc__)
    for alpha in alpha_list:
        f1_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f1, f1)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f1_opt_x}')
        print('-' * 20)

    print(f2.__doc__)
    for alpha in alpha_list:
        f2_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f2, f2)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f2_opt_x}')
        print('-' * 20)
    print('\n')


def parameter_hyper_tuning_eps(x0, alpha, eps_list):
    print("Parameter Hyper Tuning epsilon:")

    print(f1.__doc__)
    for eps in eps_list:
        f1_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f1, f1)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f1_opt_x}')
        print('-' * 20)

    print(f2.__doc__)
    for eps in eps_list:
        f2_opt_x = gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f2, f2)
        print(f'x0 = {x0}, alpha = {alpha}, eps = {eps}, x_opt = {f2_opt_x}')
        print('-' * 20)
    print('\n')


def main():
    parameter_hyper_tuning_x0(x0_list=[3, -3], alpha=0.1, eps=0.001)
    parameter_hyper_tuning_alpha(x0=3, alpha_list=[1, 0.001, 0.0001], eps=0.001)
    parameter_hyper_tuning_eps(x0=3, alpha=0.1, eps_list=[0.1, 0.01, 0.0001])


if __name__ == '__main__':
    with open('parameter_hyper_tuning_one_variable_brief.txt', 'w') as file:
        with contextlib.redirect_stdout(file):
            main()
