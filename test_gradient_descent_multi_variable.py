import gradient_descent_mul_variable as gd
import matplotlib.pyplot as plt
import numpy as np
import contextlib
from mpl_toolkits.mplot3d import Axes3D


def f_x_y(x):
    """f(x, y) = x * x + y * y"""
    return x[0] ** 2 + x[1] ** 2


def test_gradient_descent_two_variable():
    print('Testing gradient descent algorithm for two variables.')

    p0, alpha, eps =[3, 3], 0.1, 0.001

    opt_x, opt_y = gd.gradient_descent_multi_variable(p0, alpha, eps, gd.derivative_approximation_multi_variable, f_x_y)
    opt_z = f_x_y([opt_x, opt_y])

    # Create a grid of x and y values
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Compute Z = f(x, y) on the grid
    Z = f_x_y([X, Y])

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface. You can change the colormap if desired.
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
    ax.scatter(opt_x, opt_y, opt_z, color='red', s=50, label='optimal x, y')
    ax.text(opt_x, opt_y, opt_z, f'({opt_x:.2f}, {opt_y:.2f}, {opt_z:.2f})',
            fontsize=10, fontweight='bold', color='black')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Plot of f(x, y) = x² + y²')
    ax.legend()

    # Display the plot
    plt.savefig('Figure_5_f(x, y)')

    plt.close()


def main():
    test_gradient_descent_two_variable()


if __name__ == '__main__':
    main()