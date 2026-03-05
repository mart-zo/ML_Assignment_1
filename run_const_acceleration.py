import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(timesteps, positions, learn_rate, max_iter, tol=1e-6):
    p0 = 0.0
    v = 0.0
    a = 0

    for i in range(max_iter):
        # p = p0 + v*t + at²
        p = p0 + v * timesteps + (a * timesteps**2)

        # residual error: position - prediction
        error = positions - p

        # calculate partial gradient
        grad_p0 = -2 * np.sum(error)
        grad_v = -2 * np.sum(timesteps * error)
        grad_a = -2 * np.sum((timesteps**2) * error)

        # update the parameters using the defined learning rate
        p0_new = p0 - learn_rate * grad_p0
        v_new = v - learn_rate * grad_v
        a_new = a - learn_rate * grad_a

        if np.abs(p0_new - p0) < tol and np.abs(v_new - v) < tol and np.abs(a_new - a) < tol:
            print(f"stopped after {i} iterations")
            break

        p0, v, a = p0_new, v_new, a_new

    return p0, v, a

def calc_residual_error(timesteps, p_i, p0, v, a):
    # calculate the error for all 6 timesteps (np.array)
    error = (p_i - (p0 + v * timesteps + a * timesteps**2))**2
    return error


def draw(p0_x, p0_y, p0_z, v_x, v_y, v_z, a_x, a_y, a_z):
    t_line = np.linspace(0, 6, 100)

    # Calculate the predicted coordinates for the line
    x_line = p0_x + v_x * t_line + a_x * t_line ** 2
    y_line = p0_y + v_y * t_line + a_y * t_line ** 2
    z_line = p0_z + v_z * t_line + a_z * t_line ** 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0.5, 5.0])

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
    ax.set_title('Drone Trajectory: Data vs. Constant Acceleration Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('drone_trajectory_estimation_acceleration.png')


def predict_new_pos_and_draw(t_pred, p0_x, p0_y, p0_z, v_x, v_y, v_z, a_x, a_y, a_z):

        t_line = np.linspace(0, t_pred, 100)

        # Calculate the predicted coordinates for the line
        x_line = p0_x + v_x * t_line + a_x * t_line ** 2
        y_line = p0_y + v_y * t_line + a_y * t_line ** 2
        z_line = p0_z + v_z * t_line + a_z * t_line ** 2

        # Calculate the specific prediction point for t=7
        x_7 = p0_x + v_x * t_pred + a_x * t_pred ** 2
        y_7 = p0_y + v_y * t_pred + a_y * t_pred ** 2
        z_7 = p0_z + v_z * t_pred + a_z * t_pred ** 2

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([0.5, 5.0])

        ax.plot(p_x, p_y, p_z, 'b-', linewidth=2, label='Tracked Trajectory')

        # 2. Plot the original data points (red dots) [cite: 11, 12]
        ax.scatter(p_x, p_y, p_z, c='r', marker='o', s=100, label='Tracked Positions (Data)')

        # 3. Plot the Regression Curve (green dashed line) [cite: 13, 20]
        ax.plot(x_line, y_line, z_line, 'g--', linewidth=2, label='Constant Acceleration Fit')

        # 3. Plot p0 (the initial position at t=0)
        ax.scatter(p0_x, p0_y, p0_z, c='black', marker='x', s=150, label='Initial Position (p0)')

        ax.scatter(x_7, y_7, z_7, c='gold', marker='*', s=200, label=f'prediction t={t_pred}')


        # Labels and title
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Drone Trajectory: Data vs. Constant Acceleration Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('drone_trajectory_predict_t7.png')

        print(x_7, y_7, z_7)



def main():
    # Timesteps
    timesteps = np.array([1, 2, 3, 4, 5, 6])

    # 3D points
    p_x = np.array([2.00, 1.08, -0.83, -1.97, -1.31, 0.57])
    p_y = np.array([0.00, 1.68, 1.82, 0.28, -1.51, -1.91])
    p_z = np.array([1.00, 2.38, 2.49, 2.15, 2.59, 4.32])

    # To find v_x, v_y, v_z
    p0_x, v_x, a_x = gradient_descent(timesteps, p_x, 0.00001, 1000000)
    print(f"For Axis X: p0={p0_x}, v={v_x}, a={a_x}")
    p0_y, v_y, a_y = gradient_descent(timesteps, p_y, 0.00001, 1000000)
    print(f"For Axis Y: p0={p0_y}, v={v_y}, a={a_y}")
    p0_z, v_z, a_z = gradient_descent(timesteps, p_z, 0.00001, 1000000)
    print(f"For Axis Z: p0={p0_z}, v={v_z}, a={a_z}")

    error_xyz = (calc_residual_error(timesteps, p_x, p0_x, v_x, a_x) +
               calc_residual_error(timesteps, p_y, p0_y, v_y, a_y) +
               calc_residual_error(timesteps, p_z, p0_z, v_z, a_z))

    # now using our results to plot the estimated v and p0
    draw(p0_x, p0_y, p0_z, v_x, v_y, v_z, a_x, a_y, a_z)

    predict_new_pos_and_draw(7, p0_x, p0_y, p0_z, v_x, v_y, v_z, a_x, a_y, a_z)

    # now sum up all the 6 residual errors
    residual_error = np.sum(error_xyz)
    print(residual_error)


if __name__ == "__main__":
    main()