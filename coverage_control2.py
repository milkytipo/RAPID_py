import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d

# Parameters
num_robots = 10
iterations = 100
learning_rate = 0.1
x_min, x_max, y_min, y_max = 0, 10, 0, 10

# Initialize robot positions randomly
robot_positions = np.random.rand(num_robots, 2) * 10

# Function to compute the utility of a robot position
def coverage_utility(positions, grid_points):
    utility = np.zeros(len(grid_points))
    for pos in positions:
        distances = np.linalg.norm(grid_points - pos, axis=1)
        utility += np.exp(-distances)
    return np.sum(utility)

# Generate grid points to evaluate coverage
grid_resolution = 50
grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                             np.linspace(y_min, y_max, grid_resolution))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Function to find the best move using submodular greedy algorithm
def greedy_move(robot_positions, grid_points):
    new_positions = robot_positions.copy()
    for i, pos in enumerate(robot_positions):
        best_pos = pos
        best_utility = coverage_utility(robot_positions, grid_points)
        candidate_moves = pos + np.random.uniform(-0.5, 0.5, (10, 2))
        candidate_moves = np.clip(candidate_moves, [x_min, y_min], [x_max, y_max])
        for candidate in candidate_moves:
            temp_positions = robot_positions.copy()
            temp_positions[i] = candidate
            utility = coverage_utility(temp_positions, grid_points)
            if utility > best_utility:
                best_utility = utility
                best_pos = candidate
        new_positions[i] = best_pos
    return new_positions

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
    robot_positions = greedy_move(robot_positions, grid_points)
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'ro', markersize=5)
    ax.set_title(f'Coverage Control Iteration {frame}')

# Create animation
ani = FuncAnimation(fig, update, frames=iterations, repeat=False)
plt.show()
