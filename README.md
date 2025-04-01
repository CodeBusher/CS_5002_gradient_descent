# CS 5002 Gradient Descent

This repository contains the code, figures, and documentation for a gradient descent project developed for CS 5002. The project explores the gradient descent algorithm for both univariate and multivariable functions, with a focus on hyperparameter tuning and derivative approximations.

## Overview

The objective of this project is to implement and evaluate the performance of the gradient descent algorithm on various functions:

- **Univariate Functions:**  
  - $f_1(x) = x^2$
  - $f_2(x) = x^2 - 2x + 3$
- **Complex Univariate Function:**  
  - $f_3(x) = \sin(x) + \cos(\sqrt2x)$ (over the interval $0 < x < 10$)
- **Multivariable Function:**  
  - $f(x,y) = x^2 + y^2$

The project investigates:
- The convergence behavior of the algorithm.
- The sensitivity of convergence to hyperparameters, the initial point $x_0$, learning rate $\alpha$, and convergence tolerance $\epsilon$.
- The reliability of derivative approximations via finite differences compared to analytical derivatives.
