# from src import robot
import numpy as np
import matplotlib.pyplot as plt
from src.utils import geometry as geom
from src.robots import HexaArm, QuadArm

m_px_ratio = 0.0957  # meters per pixel

# Define the robot:
robot = QuadArm(theta_1_init=np.pi,
                theta_2_init=-np.pi,
                theta_4_init=-np.pi,
                d_3_init=00,
                L_1=6 / m_px_ratio,
                L_2=6 / m_px_ratio,
                L_4=5 / m_px_ratio,
                L_E=2 / m_px_ratio,
                theta_1_range=[-np.pi, np.pi],
                theta_2_range=[-np.pi, np.pi],
                theta_4_range=[-np.pi, np.pi],
                d_3_range=[-10 / m_px_ratio, 100 / m_px_ratio],
                FLOOR_Z=0,
                origin=np.array([0, 0, 0]))

# Get the robot workspace:
workspace = robot.get_work_space(no_points=21)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Workspace of QuadBot')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-3000, 3000)
ax.set_ylim(-3000, 3000)
ax.set_zlim(-0, 6000)
ax.scatter(workspace[0], workspace[1],
           workspace[2], c='b', alpha=0.1, s=0.2)
plt.show()

# Trajectory planning in joint space:
# The path to be traced is a semi-circle on the x-y plane with radius 50 and center at (0, 150, 0)
total_time = 10
t_step = 0.1
no_points = 10
height = 50  # Height of circle to be traced
target_y = 150  # Distance from origin to circle
# Get the start, via and end points
via_points = geom.get_pick_up_path_points(target_y, height, no_points)
# The orientation is constant during all via points
target_orientation = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [-1, 0, 0]])

# t, t_via_points, all_thetas, all_velocities, all_accelerations, joint_values_at_via_points = robot.traverse_via_points_in_joint_space(
#     via_points, target_orientation, total_time, t_step)

# Automatic linear trajectory planning in joint space:
target_pos = np.array([199.69, 9.549, 0])
robot.add_target(target_pos)
# t, t_via_points, all_thetas, all_velocities, all_accelerations, via_points, joint_values_at_via_points = robot.move_to_target_in_joint_space(
#     no_points, total_time, t_step)

t, t_via_points, via_points, joint_values_at_via_points, all_thetas, all_velocities, all_accelerations, end_effector_positions, end_effector_orientations = robot.move_to_target_sim(
    n_via_points=no_points, total_time=total_time, t_step=t_step
)

print(end_effector_orientations[0])

# Plot the path in cartesian space:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Pick up path in Cartesian Space')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.plot(via_points[:, 0],
        via_points[:, 1], via_points[:, 2], label='path')
ax.scatter(via_points[0, 0],
           via_points[0, 1],
           via_points[0, 2], c='g', label='Point A')
ax.scatter(via_points[1:-1, 0],
           via_points[1:-1, 1],
           via_points[1:-1, 2], c='b', label='via points')
ax.scatter(via_points[-1, 0],
           via_points[-1, 1],
           via_points[-1, 2], c='r', label='Point B')
ax.legend()
plt.show()

# Plot the path in joint space:
fig, ax = plt.subplots(4)
fig.suptitle('Pick up path in Joint Space')

# Plot the target joint positions and via points
for i in range(len(joint_values_at_via_points)):
    title = 'Translations' if i == 2 else 'Angles'
    y_label = f'$d_{i+1}$ (mm)' if i == 2 else f'$θ_{i+1}$ (rad)'
    ax[i].set_title(f'Joint {i+1} {title}')
    ax[i].set_xlabel('$t$ (s)')
    ax[i].set_ylabel(y_label)
    ax[i].scatter(t_via_points, joint_values_at_via_points[i])

# Plot the actual path of the joints
[ax[i].plot(t,
            all_thetas[i],
            label=f'linear parabolic blend path for joint{i+1}',
            c='orange')
 for i in range(len(joint_values_at_via_points))]
plt.tight_layout()
plt.show()

# Plot the actual velocities of the joints
fig, ax = plt.subplots(4)
fig.suptitle('Pick up velocities in Joint Space')
[ax[i].plot(t,
            all_velocities[i],
            label=f'Velocities for joint{i+1}',
            c='r') for i in range(len(joint_values_at_via_points))]
plt.tight_layout()
plt.show()

# Plot the actual accelerations of the joints
fig, ax = plt.subplots(4)
fig.suptitle('Pick up accelerations in Joint Space')
[ax[i].plot(t,
            all_accelerations[i],
            label=f'Accelerations for joint{i+1}',
            c='b')
 for i in range(len(joint_values_at_via_points))]
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
