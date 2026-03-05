import numpy as np
import matplotlib.pyplot as plt

def fit_poly_with_gd(t, values, degree, lr=0.001, steps=200000):
    """
    t       = times (like [1,2,3,4,5,6])
    values  = the measured values for ONE coordinate (x or y or z)
    degree  = 1 for line, 2 for quadratic
    lr      = learning rate
    steps   = number of GD iterations

    returns:
      coeff = polynomial coefficients [c0, c1, ..., c_degree]
      sse   = sum of squared errors
    """

    # --------- 1) Build the matrix Phi ---------
    # For degree 1: Phi rows are [1, t]
    # For degree 2: Phi rows are [1, t, t^2]
    Phi = []
    for ti in t:
        row = []
        for k in range(degree + 1):
            row.append(ti ** k)   # ti^0, ti^1, ti^2, ...
        Phi.append(row)
    Phi = np.array(Phi)  # shape: (N, degree+1)

    # --------- 2) Initialize coefficients ---------
    # Start with all zeros: [0,0] or [0,0,0]
    coeff = np.zeros(degree + 1)

    # --------- 3) Gradient descent loop ---------
    N = len(t)
    for _ in range(steps):

        # predicted values = Phi * coeff
        pred = Phi @ coeff

        # error for each data point
        err = pred - values

        # gradient of (1/(2N)) * sum(err^2) is: (1/N) * Phi^T * err
        grad = (Phi.T @ err) / N

        # update coefficients
        coeff = coeff - lr * grad

    # --------- 4) Compute SSE (as requested in assignment) ---------
    pred = Phi @ coeff
    sse = np.sum((pred - values) ** 2)

    return coeff, sse

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# Your 3D points (x,y,z)
P = np.array([
    [ 2.00,  0.00, 1.00],
    [ 1.08,  1.68, 2.38],
    [-0.83,  1.82, 2.49],
    [-1.97,  0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [ 0.57, -1.91, 4.32],
], dtype=float)

# times: 1..6
t = np.array([1,2,3,4,5,6], dtype=float)

# split coordinates
x = P[:,0]
y = P[:,1]
z = P[:,2]

# ---------------- (a) DEGREE 1: constant velocity ----------------
cx1, sse_x1 = fit_poly_with_gd(t, x, degree=1, lr=0.01)
cy1, sse_y1 = fit_poly_with_gd(t, y, degree=1, lr=0.01)
cz1, sse_z1 = fit_poly_with_gd(t, z, degree=1, lr=0.01)

# For degree 1: coeff = [b, v]
vx = cx1[1]
vy = cy1[1]
vz = cz1[1]

SSE1 = sse_x1 + sse_y1 + sse_z1

print("Degree 1 (constant velocity)")
print("vx, vy, vz =", vx, vy, vz)
print("SSE =", SSE1)

# ---------------- (b) DEGREE 2: constant acceleration ----------------
cx2, sse_x2 = fit_poly_with_gd(t, x, degree=2, lr=0.001)
cy2, sse_y2 = fit_poly_with_gd(t, y, degree=2, lr=0.001)
cz2, sse_z2 = fit_poly_with_gd(t, z, degree=2, lr=0.001)

SSE2 = sse_x2 + sse_y2 + sse_z2

print("\nDegree 2 (constant acceleration)")
print("SSE =", SSE2)

#acceleration meaning for degree 2:
# x(t)=c0 + c1 t + c2 t^2 -> acceleration ax = 2*c2
ax = 2 * cx2[2]
ay = 2 * cy2[2]
az = 2 * cz2[2]
print("ax, ay, az =", ax, ay, az)

# ---------------- (c) Predict t=7 using degree 2 model ----------------
t7 = 7.0
x7 = cx2[0] + cx2[1]*t7 + cx2[2]*(t7**2)
y7 = cy2[0] + cy2[1]*t7 + cy2[2]*(t7**2)
z7 = cz2[0] + cz2[1]*t7 + cz2[2]*(t7**2)

print("\nPredicted position at t=7:", x7, y7, z7)

# Plot measured points + predicted point
fig = plt.figure()
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot(x, y, z, marker="o", label="measured t=1..6")
ax3d.scatter(x7, y7, z7, marker="*", s=150, label="predicted t=7")

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.legend()
plt.show()
