import numpy as np

T = np.array([1,2,3,4,5,6])
P = np.array([
    [2,    0,    1   ],
    [1.08, 1.68, 2.38],
    [-0.83,1.82, 2.49],
    [-1.97,0.28, 2.15],
    [-1.31,-1.51,2.59],
    [0.57, -1.91,4.32]
])
One = np.array([1,1,1,1,1,1])
#create D matrix (with x as time)
D = np.column_stack((One,T))

y_x = P[:, 0] #all rows and  1st column
y_y = P[:, 1] #all rows and  2nd column
y_z = P[:, 2] #all rows and  3rd column

#looking for alpha - formula from slide 14 (lecture on linear regression)
alpha_x = np.linalg.inv(D.T @ D) @ D.T @ y_x
alpha_y = np.linalg.inv(D.T @ D) @ D.T @ y_y
alpha_z = np.linalg.inv(D.T @ D) @ D.T @ y_z

print("x0:", alpha_x[0], "Vx:", alpha_x[1])
print("y0:", alpha_y[0], "Vy:", alpha_y[1])
print("z0:", alpha_z[0], "Vz:", alpha_z[1])

# SSE
predicted_x = D @ alpha_x
predicted_y = D @ alpha_y
predicted_z = D @ alpha_z

sse_x = np.sum((y_x - predicted_x)**2)
sse_y = np.sum((y_y - predicted_y)**2)
sse_z = np.sum((y_z - predicted_z)**2)

print("sse_for_x:",sse_x)
print("sse_for_y:",sse_y)
print("sse_for_z:",sse_z)

