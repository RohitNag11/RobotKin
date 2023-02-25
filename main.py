# from src import robot
import numpy as np
import matplotlib.pyplot as plt
from src.utils import geometry as geom
from src.robots import HexaBot, QuadBot

# Define the robot:
robot = QuadBot(theta_1_init=np.pi,
                theta_2_init=-np.pi/4,
                theta_4_init=-np.pi/5,
                d_3_init=50,
                L_1=60,
                L_2=40,
                L_4=10,
                L_E=30,
                theta_1_range=[0, 2*np.pi],
                theta_2_range=[-2*np.pi, 0],
                theta_4_range=[-np.pi, np.pi],
                d_3_range=[0, 100],
                FLOOR_Z=0)

# Get the robot workspace:
workspace = robot.get_work_space(no_points=21)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Workspace of QuadBot')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_zlim(-0, 300)
ax.scatter(workspace[0], workspace[1],
           workspace[2], c='b', alpha=0.1, s=0.2)
plt.show()

# Trajectory planning in joint space:
# The path to be traced is a semi-circle on the x-y plane with radius 50 and center at (0, 150, 0)
total_time = 10
no_points = 10
height = 50  # Height of circle to be traced
target_y = 150  # Distance from origin to circle
# Get the start, via and end points
points_pos = geom.get_pick_up_path_points(target_y, height, no_points)
# The orientation is constant during all via points
target_orientation = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [-1, 0, 0]])
targets = [geom.create_t_matrix(
    target_orientation, point_pos) for point_pos in points_pos]
# Get the joint positions for each via point using inverse kinematics
all_joint_positions = np.array(
    [robot.inverse_kin(target) for target in targets]).T

# Plot the path in cartesian space:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Pick up path in Cartesian Space')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.plot(points_pos[:, 0],
        points_pos[:, 1], points_pos[:, 2], label='path')
ax.scatter(points_pos[0, 0],
           points_pos[0, 1],
           points_pos[0, 2], c='g', label='Point A')
ax.scatter(points_pos[1:-1, 0],
           points_pos[1:-1, 1],
           points_pos[1:-1, 2], c='b', label='via points')
ax.scatter(points_pos[-1, 0],
           points_pos[-1, 1],
           points_pos[-1, 2], c='r', label='Point B')
ax.legend()
plt.show()

# Use Linear functions with parabolic blends to find joint positions, velocity and accelerations during the path
# note: the functions used will have no linear part and instead will be composed of two parabolic blends
t = np.linspace(0, total_time, no_points)
t_step = 0.1  # Time step for plotting

# Initialize lists to store joint positions, velocities and accelerations for all joints
all_thetas = [[], [], [], []]
all_velocities = [[], [], [], []]
all_accelerations = [[], [], [], []]
# Populate lists with joint positions, velocities and accelerations for all joints
for j, joint_positions in enumerate(all_joint_positions):
    for i in range(0, len(joint_positions)-1):
        t_0 = t[i]
        t_f = t[i+1]
        t_f0 = t_f - t_0
        t_h = t_f0 / 2
        t_arr = np.arange(0, t_f0+t_step, t_step)
        theta_0 = joint_positions[i]
        theta_f = joint_positions[i+1]
        d_theta = theta_f - theta_0
        a = d_theta / t_h**2
        v = a * t_h
        for t_i in t_arr:
            if t_i <= t_h:
                theta_i = theta_0 + (v / t_f0) * t_i**2
                v_i = t_i * v / t_h
                a_i = v / t_h
            else:
                theta_i = theta_f - (a * t_f0**2) / 2 + \
                    a * t_f0 * t_i - (a / 2) * t_i**2
                v_i = a * t_f - a * t_i**2
                a_i = -2 * a * t_i**2
            all_thetas[j].append([t_i+t_0, theta_i])
            all_velocities[j].append([t_i+t_0, v_i])
            all_accelerations[j].append([t_i+t_0, a_i])
all_thetas = np.array(all_thetas)
all_velocities = np.array(all_velocities)
all_accelerations = np.array(all_accelerations)

# Plot the path in joint space:
fig, ax = plt.subplots(4)
fig.suptitle('Pick up path in Joint Space')

# Plot the target joint positions and via points
for i in range(len(all_joint_positions)):
    title = 'Translations' if i == 2 else 'Angles'
    y_label = f'$d_{i+1}$ (mm)' if i == 2 else f'$θ_{i+1}$ (rad)'
    ax[i].set_title(f'Joint {i+1} {title}')
    ax[i].set_xlabel('$t$ (s)')
    ax[i].set_ylabel(y_label)
    ax[i].scatter(t, all_joint_positions[i])

# Plot the actual path of the joints
[ax[i].plot(all_thetas[i][:, 0],
            all_thetas[i][:, 1],
            label=f'linear parabolic blend path for joint{i+1}',
            c='orange')
 for i in range(len(all_joint_positions))]
plt.tight_layout()
plt.show()

# Plot the actual velocities of the joints
fig, ax = plt.subplots(4)
fig.suptitle('Pick up velocities in Joint Space')
[ax[i].plot(all_velocities[i][:, 0],
            all_velocities[i][:, 1],
            label=f'Velocities for joint{i+1}',
            c='r') for i in range(len(all_joint_positions))]
plt.tight_layout()
plt.show()

# Plot the actual accelerations of the joints
fig, ax = plt.subplots(4)
fig.suptitle('Pick up accelerations in Joint Space')
[ax[i].plot(all_accelerations[i][:, 0],
            all_accelerations[i][:, 1],
            label=f'Accelerations for joint{i+1}',
            c='b')
 for i in range(len(all_joint_positions))]
plt.tight_layout()
plt.show()

# Visualise the robot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-10, 190)
colors = ['k', 'y', 'b', 'c', 'm', 'g', 'k', 'y', 'b', 'c', 'm', 'g']
for i in range(1, len(robot.joint_links[1:])+1):
    m_i = robot.joint_links[i-1]
    m_j = robot.joint_links[i]
    link_start = m_i.global_position
    link_end = m_j.global_position
    link_length = geom.get_distance_between_points(link_start, link_end)
    if link_length > 0:
        link_coords = np.array([link_start, link_end])
        name = f'link{i-1}'
        ax.plot(link_coords[:, 0],
                link_coords[:, 1],
                link_coords[:, 2],
                label=name,
                lw=10,
                color=colors[i-1],
                solid_capstyle='round',
                alpha=0.2)
[ax.quiver(m.global_position[0],
           m.global_position[1],
           m.global_position[2],
           m.global_orientation[0, 2],
           m.global_orientation[1, 2],
           m.global_orientation[2, 2],
           length=20,
           normalize=False,
           alpha=1,
           ec='r') for m in robot.joint_links]
[ax.text(m.global_position[0]+25*m.global_orientation[0, 2],
         m.global_position[1]+25*m.global_orientation[1, 2],
         m.global_position[2]+25*m.global_orientation[2, 2],
         s=f'$z_{i}$',
         c='r') for i, m in enumerate(robot.joint_links)]
[ax.quiver(m.global_position[0],
           m.global_position[1],
           m.global_position[2],
           m.global_orientation[0, 1],
           m.global_orientation[1, 1],
           m.global_orientation[2, 1],
           length=20,
           normalize=False,
           alpha=1,
           ec='g') for m in robot.joint_links]
[ax.text(m.global_position[0]+25*m.global_orientation[0, 1],
         m.global_position[1]+25*m.global_orientation[1, 1],
         m.global_position[2]+25*m.global_orientation[2, 1],
         s=f'$y_{i}$',
         c='g') for i, m in enumerate(robot.joint_links)]
[ax.quiver(m.global_position[0],
           m.global_position[1],
           m.global_position[2],
           m.global_orientation[0, 0],
           m.global_orientation[1, 0],
           m.global_orientation[2, 0],
           length=20,
           normalize=False,
           alpha=1,
           ec='b') for m in robot.joint_links]
[ax.text(m.global_position[0]+25*m.global_orientation[0, 0],
         m.global_position[1]+25*m.global_orientation[1, 0],
         m.global_position[2]+25*m.global_orientation[2, 0],
         s=f'$x_{i}$',
         c='b') for i, m in enumerate(robot.joint_links)]
[ax.scatter(m.global_position[0],
            m.global_position[1],
            m.global_position[2],
            c=colors[i],
            label=m.name) for i, m in enumerate(robot.joint_links)]

ax.legend(bbox_to_anchor=(1, 0),
          loc="lower right",
          bbox_transform=plt.gcf().transFigure)
plt.show()
