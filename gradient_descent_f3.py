import matplotlib.pyplot as plt
import numpy as np
import gradient_descent_one_variable as gd
import plot as p


def f3(x):
    """f(x) = sin(x) + cos(sqrt(2) * x)"""
    x = np.asarray(x)
    return np.sin(x) + np.cos(np.sqrt(2) * x)


def deriv_f3(f, x):
    """f(x) = sin(x) + cos(sqrt(2) * x)"""
    x = np.asarray(x)
    return np.cos(x) - np.sqrt(2) * np.sin(np.sqrt(2) * x)


def plot_f3():
    """Plot of f(x) = sin(x) + cos(sqrt(2) * x), 0 <= x <= 10"""
    plt.figure()

    x = np.linspace(0, 10, 100)
    y = f3(x)

    plt.plot(x, y, label=f3.__doc__)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f3.__doc__)
    plt.legend()

    plt.savefig('Figure_3_f3')

    plt.close()


def gradient_descent_f3():
    """Plot of f(x) = sin(x) + cos(sqrt(2) * x), 0 <= x <= 10 with optimal positions."""
    x0_list, alpha, eps = [1, 4, 5, 7], 0.1, 0.0001

    # Find the optimal x for each (x0, alpha, eps)
    f3_opt_x = []
    for x0 in x0_list:
        f3_opt_x.append(gd.gradient_descent_one_variable(x0, alpha, eps, deriv_f3, f3))

    # Print the value of x0 and its corresponding optimal x
    print(f3.__doc__)
    for i in range(4):
        print(f'x0 = {x0_list[i]}, x_opt = {f3_opt_x[i]}')
    print('-'*10)

    # Plot the results
    p.plot_opt(f3, f3_opt_x, 'Figure_4_f3_optimal_x', x_left=0, x_right=10)


def main():
    plot_f3()
    gradient_descent_f3()


if __name__ == '__main__':
    main()