import numpy as np

# Timesteps
timesteps = np.array([1, 2, 3, 4, 5, 6])

# 3D points
p_x = np.array([2.00,  1.08, -0.83, -1.97, -1.31,  0.57])
p_y = np.array([0.00,  1.68,  1.82,  0.28, -1.51, -1.91])
p_z = np.array([1.00,  2.38,  2.49,  2.15,  2.59,  4.32])

def gradient_descent(startPoint, gradient, learn_rate, max_iter, tol=0.001):
    """
    Performs gradient descent to minimize a given function.

    Parameters:
    start (float): The starting point for the algorithm.
    gradient (callable): The gradient of the function.
    learn_rate (float): The learning rate (step size).
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for stopping the algorithm.

    Returns:
    float: The point at which the function is minimized.
    """
    x = np.array(startPoint, dtype=float)  # Initialize the starting point
    for it in range(max_iter):
        diff = learn_rate * gradient(x)  # Calculate the step size
        if np.linalg.norm(diff) < tol:  # Check if the step size is smaller than the tolerance
            break  # If yes, stop the algorithm
        print("iteration =", it, "\t\tx =", "{:.5f}".format(x), "\t\tf(x) =", "{:.3f}".format(function(x)))
        x = x - diff  # Update the current point
    return x

# Define the function p(t) = v*t + p0 where the weights are v, p0
def func_const_velocity(weights, t_data, p_data):
    """
    The function to minimize.

    Parameters:
    x (float): The input value.

    Returns:
    float: The function value at x.
    """
    v, p0 = weights
    prediction = v * t_data + p0
    dv = 2 * np.sum((preds - p_data) * t_data)  # gradient function (derivatinve)

    return x**2 - 4*x + 1

# Define the gradient of the function, which is f'(x) = 2x - 4
def gradient_func(x):
    """
    The gradient of the function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The gradient value at x.
    """
    return 2*x - 4

# Run the gradient descent algorithm starting from x = 9
results_const_velocity = []
for data in [p_x, p_y, p_z]:
    res = gradient_descent([0.0, 0.0], lambda w: grad_linear(w, T, data), 0.001, 50000)
    results_const_velocity.append(res)

print(f"Const Velocity: {results_const_velocity}")


gradient_descent(9, func, gradient_func, 0.1, 100)