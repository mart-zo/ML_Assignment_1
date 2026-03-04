import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(P, t, learning_rate = 0.001, max_iterations = 100000, tolerance = 0.001, model = 1):
    w1 = np.zeros(3)
    # b = np.zeros(3)
    b = [2.00, 0.00, 1.00]
    w2 = np.zeros(3)
    t_col = t.reshape(-1, 1)
    n = len(t_col)
    if model == 1:  # Constant Speed
        for epoch in range(max_iterations):
            P_hat = w1 * t_col + b
            grad_w1 = -2 * np.sum(t_col * (P - P_hat), axis=0)
            grad_b = -2 * np.sum(P - P_hat, axis=0)
            w1 = w1 - learning_rate * grad_w1
            # b = b - learning_rate * grad_b
    elif model == 2: #Constant Acceleration
        for epoch in range(max_iterations):
            P_hat = b + w1 * t_col + w2 * t_col**2
            grad_w1 = -2/n * np.sum(t_col*(P - P_hat), axis=0)
            grad_w2 = -2/n * np.sum(t_col**2 * (P - P_hat), axis=0)
            grad_b = -2/n * np.sum(P - P_hat, axis=0)
            w1 = w1 - learning_rate * grad_w1
            w2 = w2 - learning_rate * grad_w2
            # b = b - learning_rate * grad_b

    P_hat = b + w1 * t_col + w2 * t_col**2
    res_error = np.sum((P - P_hat) ** 2)
    if (model == 2):
        print("Estimated acceleration (ax, ay, az):", w2 * 2)
    print("Estimated velocity (vx, vy, vz):", w1)
    print("Estimated initial position (bx, by, bz):", b)
    print("Residual sum-of-squares error:", res_error)

    return P_hat

def main():

    P = np.array([
        [2.00, 0.00, 1.00],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ])
    t = np.linspace(1, 6, 6)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="red")
    ax.plot(P[:, 0], P[:, 1], P[:, 2])
    for i in range(len(t)):
        ax.text(P[i, 0], P[i, 1], P[i, 2], f'{t[i]}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # p lt.show()
    P_hat = gradient_descent(P, t)
    ax.plot(P_hat[:, 0], P_hat[:, 1], P_hat[:, 2])
    plt.show()

if __name__ == '__main__':
    main()
