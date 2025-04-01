import matplotlib.pyplot as plt
import numpy as np


def plot_opt(f, x_opt, filename, x_left=-5, x_right=5, x_num=100):
    """
    Plot the function f(x) over a specified interval and mark the optimal point where f(x) is local minimum.
    :param f: The function to be plotted
    :param x_opt: The optimal x value, where f(x) is local minimum.
    :param filename: The file path where the plot will be saved.
    :param x_left: The left endpoint of the x range
    :param x_right: The right endpoint of the x range
    :param x_num: The number of points in the x grid
    """

    # create a new figure
    plt.figure()

    # Generate an array of x values from x_left to x_right with x_num points and calculate corresponding y
    x = np.linspace(x_left, x_right, x_num)
    y = f(x)

    label_text = f.__doc__ if f.__doc__ is not None else 'f(x)'
    plt.plot(x, y, label=label_text)

    if not isinstance(x_opt, list):
        x_opt = [x_opt]
    y_opt = [f(x) for x in x_opt]
    plt.scatter(x_opt, y_opt, color='red', label='Optimal x')
    for xi, yi in zip(x_opt, y_opt):
        plt.text(xi-1, yi+1, f'({xi:.2f}, {yi:.2f})', fontsize=10, fontweight='bold', color='black')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Plot with Optimal Position')
    plt.legend()

    plt.savefig(filename)

    plt.close()
