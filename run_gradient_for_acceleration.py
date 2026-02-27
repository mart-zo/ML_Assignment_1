import numpy as np

#data from the assignment - to get sse later = a function we want to minimize using gradient descent:
T = np.array([1, 2, 3, 4, 5, 6])
P = np.array([
    [2, 0, 1],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
    ])
One = np.array([1, 1, 1, 1, 1, 1])

#create D_a matrix (with x as time) - for acceleration
D_a = np.column_stack((One, T, T**2))
y_x = P[:, 0]  #all rows and  1st column (the 6 measured x-positions)
y_y = P[:, 1]  #all rows and  2nd column (y positions)
y_z = P[:, 2]  #all rows and  3rd column (z positions)

# define gradient descent function (to be used later) - changed one thing from the original fun since aplha is a vector
def gradient_descent(start, function, gradient, learn_rate, max_iter, tol=1e-6):
    """
    Performs gradient descent to minimize a given function.

    Parameters:
    start (float): The starting point for the algorithm.
    function (callable): The function to minimize.
    gradient (callable): The gradient of the function.
    learn_rate (float): The learning rate (step size).
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for stopping the algorithm.

    Returns:
    float: The point at which the function is minimized.
    """

    x = start  # Initialize the starting point
    for it in range(max_iter):
        diff = learn_rate * gradient(x)  # Calculate the step size
        if np.linalg.norm(diff) < tol:  # Check if the step size is smaller than the tolerance - chnaged to np.linalg.norm to get length of a vector
            print(f"Converged after {it} iterations")
            break


        x = x - diff  # Update the current point
    return x

T_norm = (T - T.mean()) / T.std()  # scale T to have mean=0, std=1

#task 2.2b - now we have p(t) = a0 +a1t + a2t^2
def sse_x_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 *T_norm  + a2 * T_norm**2
    return np.sum((y_x - predicted)**2)

def sse_y_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 * T_norm + a2 * T_norm**2
    return np.sum((y_y - predicted)**2)

def sse_z_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 * T_norm + a2 * T_norm**2
    return np.sum((y_z - predicted)**2)

def grad_x_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 * T_norm + a2 * T_norm**2
    da0 = -2 * np.sum(y_x - predicted)
    da1 = -2 * np.sum(T_norm * (y_x - predicted))
    da2 = -2 * np.sum(T_norm**2 * (y_x - predicted))
    return np.array([da0, da1, da2])

def grad_y_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 * T_norm + a2 * T_norm**2
    da0 = -2 * np.sum(y_y - predicted)
    da1 = -2 * np.sum(T_norm * (y_y - predicted))
    da2 = -2 * np.sum(T_norm**2 * (y_y - predicted))
    return np.array([da0, da1, da2])

def grad_z_a(alpha):
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    predicted = a0 + a1 * T_norm + a2 * T_norm**2
    da0 = -2 * np.sum(y_z - predicted)
    da1 = -2 * np.sum(T_norm * (y_z - predicted))
    da2 = -2 * np.sum(T_norm**2 * (y_z - predicted))
    return np.array([da0, da1, da2])


# Run the gradient descent algorithm starting from x = 0 - for acceleration

alpha_x_a = gradient_descent(np.zeros(3), sse_x_a, grad_x_a, learn_rate=0.001, max_iter=1000000)

alpha_y_a = gradient_descent(np.zeros(3), sse_y_a, grad_y_a, learn_rate=0.001, max_iter=1000000)

alpha_z_a = gradient_descent(np.zeros(3), sse_z_a, grad_z_a, learn_rate=0.001, max_iter=1000000)

#results for task 2.2 (b)
print(f"x: start={alpha_x_a[0]},  velocity={alpha_x_a[1]},  acceleration={alpha_x_a[2]}")
print(f"y: start={alpha_y_a[0]},  velocity={alpha_y_a[1]},  acceleration={alpha_y_a[2]}")
print(f"z: start={alpha_z_a[0]},  velocity={alpha_z_a[1]},  acceleration={alpha_z_a[2]}")
print(f"\nVelocity vector: [{alpha_x_a[1]}, {alpha_y_a[1]}, {alpha_z_a[1]}]")
print(f"Acceleration vector: [{alpha_x_a[2]}, {alpha_y_a[2]}, {alpha_z_a[2]}]")

print(f"SSE x: {sse_x_a(alpha_x_a)}")
print(f"SSE y: {sse_y_a(alpha_y_a)}")
print(f"SSE z: {sse_z_a(alpha_z_a)}")
print(f"SSE total: {sse_x_a(alpha_x_a) + sse_y_a(alpha_y_a) + sse_z_a(alpha_z_a)}")