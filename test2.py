import numpy as np

# 1. Data Setup from the image
# T = [1, 2, 3, 4, 5, 6]
# P = [X_coords, Y_coords, Z_coords]
times = np.array([1, 2, 3, 4, 5, 6])
positions = np.array([
    [2, 1.08, -0.83, -1.97, -1.31, 0.57],  # X
    [0, 1.68, 1.82, 0.28, -1.51, -1.91],  # Y
    [1, 2.38, 2.49, 2.15, 2.59, 4.32]  # Z
])


def gradient_descent(start, t, p_obs, learn_rate, max_iter, tol=1e-7):
    """
    Performs gradient descent to minimize Sum of Squares Error.
    'start' is a vector [velocity, intercept]
    """
    w = np.array(start, dtype=float)

    for it in range(max_iter):
        # 1. Prediction: p = v*t + p0
        p_pred = w[0] * t + w[1]

        # 2. Calculate Gradient of RSS:
        # Error = (p_pred - p_obs)
        error = p_pred - p_obs

        # Gradient with respect to v and p0
        grad_v = 2 * np.sum(error * t)
        grad_p0 = 2 * np.sum(error)
        grad = np.array([grad_v, grad_p0])

        # 3. Update (The core of your original logic)
        diff = learn_rate * grad
        if np.linalg.norm(diff) < tol:
            break

        w = w - diff

    return w


# Solve for each dimension (X, Y, Z)
final_velocities = []
total_residual_error = 0

print("Dimension | Velocity (v) | Intercept (p0)")
print("-" * 40)

for i in range(3):
    # Start with [0, 0] for velocity and intercept
    # Lower learn_rate (0.001) helps stability with Sum of Squares
    best_fit = gradient_descent([0.0, 0.0], times, positions[i], 0.005, 10000)

    v, p0 = best_fit
    final_velocities.append(v)

    # Calculate RSS for this dimension
    error_sq = np.sum(((v * times + p0) - positions[i]) ** 2)
    total_residual_error += error_sq

    dim_name = ['X', 'Y', 'Z'][i]
    print(f"    {dim_name}     |   {v:7.4f}   |   {p0:7.4f}")

print("-" * 40)
print(f"Velocity Vector v = ({final_velocities[0]:.3f}, {final_velocities[1]:.3f}, {final_velocities[2]:.3f})")
print(f"Total Residual Error (SSE) = {total_residual_error:.4f}")