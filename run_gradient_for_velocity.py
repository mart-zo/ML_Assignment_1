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
#create D matrix (with x as time) - for velocity
D = np.column_stack((One, T))

y_x = P[:, 0]  #all rows and  1st column (the 6 measured x-positions)
y_y = P[:, 1]  #all rows and  2nd column (y positions)
y_z = P[:, 2]  #all rows and  3rd column (z positions)


# define gradient descent function (to be used later) - changed one thing from the original fun since aplha is a vector
def gradient_descent(start, function, gradient, learn_rate, max_iter, tol=0.001):
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
            break  # If yes, stop the algorithm
        print("iteration =", it, "\t\tx =", x, "\t\tf(x) =", "{:.3f}".format(function(x)))

        x = x - diff  # Update the current point
    return x

#p(t) = a0 +a1t - we get sse from this
#get sse of each dimension -   - sse is the function to minimize in gd
def sse_x(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    return np.sum((y_x - predicted)**2)

def sse_y(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    return np.sum((y_y - predicted)**2)

def sse_z(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    return np.sum((y_z - predicted)**2)

# Define the gradient of the function -  gradient of sse

def grad_x(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    da0 = -2 * np.sum(y_x - predicted)        # ∂E/∂a0
    da1 = -2 * np.sum(T * (y_x - predicted))    # ∂E/∂a1
    return np.array([da0, da1])

def grad_y(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    da0 = -2 * np.sum(y_y - predicted)
    da1 = -2 * np.sum(T * (y_y - predicted))
    return np.array([da0, da1])

def grad_z(alpha):
    a0, a1 = alpha[0], alpha[1]
    predicted = a0 + a1 * T
    da0 = -2 * np.sum(y_z - predicted)
    da1 = -2 * np.sum(T * (y_z - predicted))
    return np.array([da0, da1])

# Run the gradient descent algorithm starting from x = 0 - for velocity
print("X dimension")
alpha_x = gradient_descent(np.zeros(2), sse_x, grad_x, learn_rate=0.001, max_iter=10000)

print("\nY dimension")
alpha_y = gradient_descent(np.zeros(2), sse_y, grad_y, learn_rate=0.001, max_iter=10000)

print("\nZ dimension")
alpha_z = gradient_descent(np.zeros(2), sse_z, grad_z, learn_rate=0.001, max_iter=10000)


#results for task 2.2 (a)
print(f"x: start={alpha_x[0]},  velocity={alpha_x[1]}")
print(f"y: start={alpha_y[0]},  velocity={alpha_y[1]}")
print(f"z: start={alpha_z[0]},  velocity={alpha_z[1]}")
print(f"\nVelocity vector: [{alpha_x[1]}, {alpha_y[1]}, {alpha_z[1]}]")

print(f"SSE x: {sse_x(alpha_x)}")
print(f"SSE y: {sse_y(alpha_y)}")
print(f"SSE z: {sse_z(alpha_z)}")
print(f"SSE total: {sse_x(alpha_x) + sse_y(alpha_y) + sse_z(alpha_z)}")



