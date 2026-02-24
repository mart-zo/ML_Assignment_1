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

# analytical solution first to know what we're aiming for
#create D_a matrix (with x as time) - for acceleration
D_a = np.column_stack((One, T, T**2))
y_x = P[:, 0]  #all rows and  1st column (the 6 measured x-positions)
y_y = P[:, 1]  #all rows and  2nd column (y positions)
y_z = P[:, 2]  #all rows and  3rd column (z positions)



alpha_x_a = np.linalg.inv(D_a.T @ D_a) @ D_a.T @ y_x
alpha_y_a = np.linalg.inv(D_a.T @ D_a) @ D_a.T @ y_y
alpha_z_a = np.linalg.inv(D_a.T @ D_a) @ D_a.T @ y_z

print(f"x: start={alpha_x_a[0]},  velocity={alpha_x_a[1]},  acceleration={alpha_x_a[2]}")
print(f"y: start={alpha_y_a[0]},  velocity={alpha_y_a[1]},  acceleration={alpha_y_a[2]}")
print(f"z: start={alpha_z_a[0]},  velocity={alpha_z_a[1]},  acceleration={alpha_z_a[2]}")

print(f"SSE x: {np.sum((y_x - D_a @ alpha_x_a)**2)}")
print(f"SSE y: {np.sum((y_y - D_a @ alpha_y_a)**2)}")
print(f"SSE z: {np.sum((y_z - D_a @ alpha_z_a)**2)}")
print(f"SSE total: {np.sum((y_x - D_a @ alpha_x_a)**2) + np.sum((y_y - D_a @ alpha_y_a)**2) + np.sum((y_z - D_a @ alpha_z_a)**2)}")