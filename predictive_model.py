import gradient_descent_mul_variable as gd
import matplotlib.pyplot as plt
import numpy as np

n = 20

patient_x = [None] * n
patient_y = [None] * n


def read_data(filename):
    """Read the data from the given file."""
    global patient_x, patient_y
    with open(filename, "r") as file:
        for i in range(n):
            line = file.readline()
            cholesterol, blood_pressure = line.split()
            patient_x[i] = float(cholesterol)
            patient_y[i] = float(blood_pressure)


def g_a_b(para):
    """g(a, b) = (ax+b-y)^2"""
    a, b = para
    result = 0
    for i in range(n):
        result += (a * patient_x[i] + b - patient_y[i]) ** 2
    return result


def f(a, b, x):
    """f(x) = a * x + b"""
    return a * x + b


def g1_a_b_c(para):
    """g1(a, b, c) = (ax*x + b*x + c - y)^2"""
    a, b, c = para
    result = 0
    for i in range(n):
        result += (a * patient_x[i] * patient_x[i] + b * patient_x[i] + c - patient_y[i]) ** 2
    return result


def f1(a, b, c, x):
    """f(x) = a * x * x + b * x + c"""
    return a * x * x + b * x + c


def get_initial_guess():
    # Convert lists to numpy arrays for easier calculation
    x_arr = np.array(patient_x)
    y_arr = np.array(patient_y)

    # Estimate slope using the ratio of ranges
    a_init = (max(y_arr) - min(y_arr)) / (max(x_arr) - min(x_arr))

    # Estimate intercept to make the line pass through the center of the data
    b_init = np.mean(y_arr) - a_init * np.mean(x_arr)

    return [a_init, b_init]


def get_optimal_parameters():
    """Calculate optimal parameters directly"""
    x_arr = np.array(patient_x)
    y_arr = np.array(patient_y)

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sum((x_arr - x_mean) ** 2)

    a_optimal = numerator / denominator
    b_optimal = y_mean - a_optimal * x_mean

    return [a_optimal, b_optimal]


def calc_r_square(a, b):
    """Illustrate the performance of linear predictive model."""
    x_arr = np.array(patient_x)
    y_arr = np.array(patient_y)

    y_mean = np.mean(y_arr)
    y_pred = a * x_arr + b

    # Calculate sums of squares
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_mean) ** 2)

    # Calculate r_square
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def plot_linear_regression(opt_a, opt_b, filename):
    """Plot the linear regression function and patient data"""
    plt.figure()

    x_left = int(min(patient_x))
    x_right = int(max(patient_x))
    f_x = np.linspace(x_left, x_right, 2)
    f_y = f(opt_a, opt_b, f_x)

    r_square = calc_r_square(opt_a, opt_b)

    plt.plot(f_x, f_y, '-', label=f"f(x) = {opt_a:.4f} * x + {opt_b:.4f}, R^2 = {r_square:.4f}",
             color='Red', linewidth=2)
    plt.scatter(patient_x, patient_y, label="Patient data", color='Blue', alpha=0.7)

    plt.xlabel('Total Cholesterol Level (mmol/L)')
    plt.ylabel('Diastolic Blood Pressure (mmHg)')
    plt.title('Blood Pressure and Cholesterol Level')
    plt.legend()

    plt.savefig(filename)

    plt.close()


def linear_regression(datafile):
    """Find the parameters a, b for linear regression function f(x) = a * x + b"""
    read_data(datafile)

    p0 = get_initial_guess()
    print("Initial guess: ", p0)

    alpha = 1e-6
    eps = 1e-9
    opt_a, opt_b = gd.gradient_descent_multi_variable(p0, alpha, eps, gd.derivative_approximation_multi_variable,
                                                      g_a_b, iter_max=10000000)
    # Calculate the analytical solution for testing.
    ana_a, ana_b = get_optimal_parameters()
    print(f"Analytical solution: a = {ana_a}, b = {ana_b:}")
    return opt_a, opt_b


def dataset_scaling():
    global patient_x, patient_y
    x_mean, x_std = np.mean(patient_x), np.std(patient_x)
    y_mean, y_std = np.mean(patient_y), np.std(patient_y)
    for i in range(n):
        patient_x[i] = (patient_x[i] - x_mean) / x_std
        patient_y[i] = (patient_y[i] - y_mean) / y_std


def linear_regression_dataset_scaling(datafile):
    read_data(datafile)
    dataset_scaling()
    # After dataset scaling, the initial guess could be [0, 0].
    p0 = [0, 0]
    alpha = 0.01
    eps = 1e-5
    opt_a, opt_b = gd.gradient_descent_multi_variable(p0, alpha, eps, gd.derivative_approximation_multi_variable,
                                                      g_a_b, iter_max=10000)
    read_data(datafile)
    opt_a *= np.std(patient_y) / np.std(patient_x)
    opt_b = np.mean(patient_y) - opt_a * np.mean(patient_x) + opt_b * np.std(patient_y)
    return opt_a, opt_b


def non_linear():
    # Data preparation
    read_data("data_chol_dias_pressure_non_lin.txt")
    dataset_scaling()

    # Get the normalized a, b and c based on dataset scaling
    p0 = [0, 0, 0]
    alpha = 0.01
    eps = 1e-6
    a, b, c = gd.gradient_descent_multi_variable(p0, alpha, eps, gd.derivative_approximation_multi_variable,
                                                 g1_a_b_c, iter_max=10000)

    # Get the optimal a, b and c based on the normalized a, b and c
    read_data("data_chol_dias_pressure_non_lin.txt")
    x_arr = np.array(patient_x)
    y_arr = np.array(patient_y)
    x_mean, x_std = np.mean(x_arr), np.std(x_arr)
    y_mean, y_std = np.mean(y_arr), np.std(y_arr)

    opt_a = a * (y_std / (x_std ** 2))
    opt_b = b * (y_std / x_std) - 2 * opt_a * x_mean
    opt_c = y_std * c + y_mean - opt_a * x_mean ** 2 - opt_b * x_mean

    # Calculate r^2
    y_pred = f1(opt_a, opt_b, opt_c, x_arr)
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_mean) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    # Plot
    plt.figure()

    x_left = int(min(patient_x))
    x_right = int(max(patient_x))
    f_x = np.linspace(x_left, x_right, 100)
    f_y = f1(opt_a, opt_b, opt_c, f_x)

    plt.plot(f_x, f_y, '-', label=f"f(x) = {opt_a:.5f} * x^2 {opt_b:.5f} * x + {opt_c:.5f}\nR^2 = {r_squared:.4f}",
             color='Red', linewidth=2)
    plt.scatter(patient_x, patient_y, label="Patient data", color='Blue', alpha=0.7)

    plt.xlabel('Total Cholesterol Level (mmol/L)')
    plt.ylabel('Diastolic Blood Pressure (mmHg)')
    plt.title('Blood Pressure and Cholesterol Level')
    plt.legend()

    plt.savefig('Part_2_Figure_4_non_linear_data_linear_regression')

    plt.close()


if __name__ == '__main__':
    # Linear data
    opt_a, opt_b = linear_regression("data_chol_dias_pressure.txt")
    plot_linear_regression(opt_a, opt_b, 'Part_2_Figure_1_linear_regression')

    # Linear data with dataset scaling
    opt_a, opt_b = linear_regression_dataset_scaling("data_chol_dias_pressure.txt")
    plot_linear_regression(opt_a, opt_b, 'Part_2_Figure_2_linear_regression_dataset_scaling')

    # Non-linear data
    opt_a, opt_b = linear_regression_dataset_scaling("data_chol_dias_pressure_non_lin.txt")
    plot_linear_regression(opt_a, opt_b, 'Part_2_Figure_3_non_linear_data_linear_regression')
    non_linear()