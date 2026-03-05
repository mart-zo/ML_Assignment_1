import numpy as np
import matplotlib.pyplot as plt

#positions
P = np.array([
    [2, 0, 1],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
])

#times
T = np.array([1, 2, 3, 4, 5, 6])

#getting the x, y, z coordinates
x = P[:, 0]
y = P[:, 1]
z = P[:, 2]

#3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0.5, 5)

#the trajectory (line connecting the position points)
ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')

#plotting the points
ax.scatter(x, y, z, c='r', marker='o', s=100, label='Tracked positions')

#labels for each point
for i in range(len(T)):
    ax.text(x[i], y[i], z[i], f'  t{T[i]}', fontsize=10)


#Labels and title
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('Drone Trajectory', fontsize=14, fontweight='bold')


ax.legend()


ax.grid(True)


plt.tight_layout()
plt.savefig('drone_trajectory.png')
plt.show()
