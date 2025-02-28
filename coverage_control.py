import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d

# Parameters
num_robots = 50
iterations = 100
learning_rate = 0.1
x_min, x_max, y_min, y_max = 0, 10, 0, 10

# Initialize robot positions randomly
robot_positions = np.random.rand(num_robots, 2) * 10

# Function to compute centroid of each Voronoi cell
def compute_centroids(voronoi, robot_positions):
    centroids = []
    for i, point in enumerate(robot_positions):
        region_index = voronoi.point_region[i]
        region = voronoi.regions[region_index]
        if -1 in region or len(region) == 0:
            centroids.append(point)
            continue
        vertices = voronoi.vertices[region]
        centroid = np.mean(vertices, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

# Set up plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
points_plot, = ax.plot([], [], 'ro', markersize=5)

# Animation update function
def update(frame):
    global robot_positions
    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    vor = Voronoi(robot_positions)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k', line_width=1, line_alpha=0.6)
    centroids = compute_centroids(vor, robot_positions)
    robot_positions += learning_rate * (centroids - robot_positions)
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'ro', markersize=5)
    ax.set_title(f'Coverage Control Iteration {frame}')

# Create animation
ani = FuncAnimation(fig, update, frames=iterations, repeat=False)
plt.show()
