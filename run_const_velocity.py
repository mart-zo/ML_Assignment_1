import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(timesteps, positions, learn_rate, max_iter, tol=1e-6):
    # goal: minimize the residual error(sum of the squared differences between
    # the observed positions and the models predictions)
    # two variables: p0 and v

    p0 = 0.0
    v = 0.0

    for i in range(max_iter):
        # p = p0 + v*t
        p = p0 + v * timesteps

        # residual error: position - prediction
        error = positions - p

        # calculate partial gradient
        grad_p0 = -2 * np.sum(error)
        grad_v = -2 * np.sum(timesteps * error)

        # update the parameters using the defined learning rate
        p0_new = p0 - learn_rate * grad_p0
        v_new = v - learn_rate * grad_v

        if np.abs(p0_new - p0) < tol and np.abs(v_new - v) < tol:
            print(f"stopped after {i} iterations")
            break

        p0, v = p0_new, v_new

    return p0, v


# Timesteps
timesteps = np.array([1, 2, 3, 4, 5, 6])

# 3D points
p_x = np.array([2.00,  1.08, -0.83, -1.97, -1.31,  0.57])
p_y = np.array([0.00,  1.68,  1.82,  0.28, -1.51, -1.91])
p_z = np.array([1.00,  2.38,  2.49,  2.15,  2.59,  4.32])

# To find v_x, v_y, v_z
p0_x, v_x = gradient_descent(timesteps, p_x, 0.001, 10000)
print(f"For Axis X: p0={p0_x}, v={v_x}")
p0_y, v_y = gradient_descent(timesteps, p_y, 0.001, 10000)
print(f"For Axis Y: p0={p0_y}, v={v_y}")
p0_z, v_z = gradient_descent(timesteps, p_z, 0.001, 10000)
print(f"For Axis Z: p0={p0_z}, v={v_z}")


def calc_residual_error(timesteps, p_i, p0, v):
    # calculate the error for all 6 timesteps (np.array)
    error = (p_i - (p0 + v * timesteps))**2
    return error

error_xyz = (calc_residual_error(timesteps, p_x, p0_x, v_x) +
               calc_residual_error(timesteps, p_y, p0_y, v_y) +
               calc_residual_error(timesteps, p_z, p0_z, v_z))

# now sum up all the 6 residual errors
residual_error = np.sum(error_xyz)
print(residual_error)


def draw(p0_x, p0_y, p0_z, v_x, v_y, v_z):
    # Create a range of time for the smooth line (from start p0 at t=0 to t=6)
    t_line = np.linspace(0, 6, 100)

    # Calculate the predicted coordinates for the line
    x_line = p0_x + v_x * t_line
    y_line = p0_y + v_y * t_line
    z_line = p0_z + v_z * t_line

    # Calculate the specific start point (p0) at t=0
    p0_coords = (p0_x, p0_y, p0_z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_x, p_y, p_z, 'b-', linewidth=2, label='Tracked Trajectory')

    # 1. Plot the actual tracked points (from the assignment)
    ax.scatter(p_x, p_y, p_z, c='r', marker='o', s=100, label='Tracked Positions (Data)')

    # 2. Plot the Regression Line (Visualizing velocity and fit)
    ax.plot(x_line, y_line, z_line, 'g--', linewidth=2, label='Linear Regression Fit (v)')

    # 3. Plot p0 (the initial position at t=0)
    ax.scatter(p0_x, p0_y, p0_z, c='black', marker='x', s=150, label='Initial Position (p0)')

    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Drone Trajectory: Data vs. Constant Velocity Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('drone_trajectory_estimation.png')

# now using our results to plot the estimated v and p0
draw(p0_x, p0_y, p0_z, v_x, v_y, v_z)
