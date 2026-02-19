import numpy as np
import matplotlib.pyplot as plt
import math

P = np.array([
    [ 2.00,  0.00, 1.00],
    [ 1.08,  1.68, 2.38],
    [-0.83,  1.82, 2.49],
    [-1.97,  0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [ 0.57, -1.91, 4.32],
])

x, y, z = P[:, 0], P[:, 1], P[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, marker="o")
ax.scatter(x, y, z, s=50)

for i, (xi, yi, zi) in enumerate(P, start=1):
    ax.text(xi, yi, zi, f"t={i}", fontsize=9)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Tracked 3D trajectory (1 Hz)")

plt.show()