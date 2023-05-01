from src.mobile_robot_base_classes import DiffDriveRobot
from .quad_arm import QuadArm
from src.utils.path import Path
from src.utils import geometry as geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import cv2


class DiffDriveQuadBot(DiffDriveRobot):
    def __init__(self,
                 width: float,
                 chassis_height: float,
                 wheel_offset: float,
                 wheel_radius: float,
                 max_rpm: float,
                 global_position_init: list = None,
                 global_orientation_init: float = None):
        super().__init__(name='diff_drive_quadbot',
                         width=width,
                         chassis_height=chassis_height,
                         wheel_offset=wheel_offset,
                         wheel_radius=wheel_radius,
                         max_rpm=max_rpm)
        if global_position_init:
            self.set_global_position(global_position_init)
        if global_orientation_init:
            self.set_global_orientation(global_orientation_init)

    def add_quad_arm(self, quad_arm: QuadArm):
        self.quad_arm = quad_arm

    def add_arm_target(self, target: list[float]):
        self.target_global_position = np.array(target)
        # self.target_local_position = self.get_target_pos_in_robot_frame(target)
        self.target_local_orientation = self.quad_arm.orientation

    def __plot_path(self, ax, path: Path):
        for seg in path.segments:
            if seg.type == 'linear':
                ax.plot([seg.start_pos[0], seg.end_pos[0]],
                        [seg.start_pos[1], seg.end_pos[1]],
                        ls='--',
                        lw=1,
                        alpha=0.5,
                        color='cyan')
            elif seg.type == 'circular':
                # plot arc from start to end
                thetas = np.linspace(seg.start_theta, seg.end_theta, 100)
                alphas = thetas - np.pi / 2 * seg.direction
                center_x, center_y = seg.center
                x = center_x + seg.radius * np.cos(alphas)
                y = center_y + seg.radius * np.sin(alphas)
                ax.plot(x, y,
                        ls='--',
                        lw=1,
                        alpha=0.5,
                        color='cyan')

    def __animate_robot_path_2d(self, t, global_positions, global_orientations, trace_path: Path = None, bg_img_path=None, left_wheel_speeds=None, right_wheel_speeds=None, bg_img_scale=1):
        # Remove the third column (z values) from global_positions
        global_positions = global_positions[:, :2]

        ax_speeds = None
        if left_wheel_speeds.any() and right_wheel_speeds.any():
            # Create two subplots - one for the robot path and one for the wheel speeds
            fig, (ax, ax_speeds) = plt.subplots(
                nrows=2, ncols=1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax = plt.subplots()

        if bg_img_path:
            img = plt.imread(bg_img_path)
            # img = mpimg.imread(bg_img_path)
            img_height, img_width = img.shape[:2]
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
            ax.imshow(img, zorder=0)
        else:
            # Plot settings
            x_dim = np.max(global_positions[:, 0]) - \
                np.min(global_positions[:, 0])
            y_dim = np.max(global_positions[:, 1]) - \
                np.min(global_positions[:, 1])
            plot_dim = max(x_dim, y_dim)

            x_lims = np.array([np.min(global_positions[:, 0]) - 0.1 * x_dim,
                               np.max(global_positions[:, 0]) + 0.1 * x_dim])
            y_lims = np.array([np.min(global_positions[:, 1]) - 0.1 * y_dim,
                               np.max(global_positions[:, 1]) + 0.1 * y_dim])

            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Robot Path and Orientation")
        # Update the x and y axis labels with the scaled values
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        ax.set_xticklabels([f"{x * bg_img_scale:.1f}" for x in x_ticks])
        ax.set_yticklabels([f"{y * bg_img_scale:.1f}" for y in y_ticks])

        if trace_path:
            trace_line = self.__plot_path(ax, trace_path)

        segments = np.stack(
            [global_positions[:-1], global_positions[1:]], axis=1)
        path_lc = LineCollection(segments, cmap='Blues', lw=2, zorder=1)
        path_lc.set_array(np.linspace(0, 1, len(t) - 1))
        ax.add_collection(path_lc)

        arrow_length = np.linalg.norm(
            global_positions[-1] - global_positions[0]) * 0.07
        arrow = FancyArrowPatch((0, 0), (1, 1), arrowstyle="->",
                                lw=2, color="red", mutation_scale=10, zorder=2)
        ax.add_patch(arrow)

        # Custom legend
        orientation_handle = Line2D([], [], linestyle="-",
                                    color="red", linewidth=2)
        target_handle = Line2D([], [], linestyle="--",
                               color="cyan", alpha=0.5, linewidth=1)
        actual_handle = LineCollection([], linestyle="-",
                                       cmap='Blues', lw=2)

        legend_handles = [orientation_handle, target_handle, actual_handle]
        legend_labels = ['Orientation', 'Target Path', 'Actual Path']
        ax.legend(legend_handles, legend_labels)

        if ax_speeds:
            # Set up the wheel speeds subplot
            ax_speeds.set_title("Left and Right Wheel Speeds")
            ax_speeds.set_xlabel("Time (s)")
            ax_speeds.set_ylabel("Wheel Speed (rad/s)")
            ax_speeds.set_xlim(0, t[-1])
            # ax_speeds.set_ylim(np.min([left_wheel_speeds, right_wheel_speeds])
            #                    * 1.1, np.max([left_wheel_speeds, right_wheel_speeds]) * 1.1)
            left_wheel_speed_lims = np.array(
                [np.min(left_wheel_speeds), np.max(left_wheel_speeds)])
            right_wheel_speed_lims = np.array(
                [np.min(right_wheel_speeds), np.max(right_wheel_speeds)])
            wheel_speed_lims = np.array(
                [np.min([left_wheel_speed_lims, right_wheel_speed_lims]), np.max([left_wheel_speed_lims, right_wheel_speed_lims])])
            wheel_speed_range = wheel_speed_lims[1] - wheel_speed_lims[0]
            ax_speeds.set_ylim(
                (wheel_speed_lims[0] - 0.1 * wheel_speed_range, wheel_speed_lims[1] + 0.1 * wheel_speed_range))

            left_wheel_line, = ax_speeds.plot(
                [100, 500], [400, 1000], label="Left Wheel Speed")
            right_wheel_line, = ax_speeds.plot(
                [], [], label="Right Wheel Speed")
            ax_speeds.legend()

        def update(frame):
            end_x = global_positions[frame, 0] + \
                np.cos(global_orientations[frame]) * arrow_length
            end_y = global_positions[frame, 1] + \
                np.sin(global_orientations[frame]) * arrow_length
            arrow.set_positions(global_positions[frame], (end_x, end_y))
            path_lc.set_segments(segments[:frame + 1])

            updated_artists = (arrow, path_lc)

            if ax_speeds:
                new_t = t[:frame + 1]
                new_left_wheel_speeds = left_wheel_speeds[:frame + 1]
                new_right_wheel_speeds = right_wheel_speeds[:frame + 1]
                left_wheel_line.set_data(new_t, new_left_wheel_speeds)
                right_wheel_line.set_data(new_t, new_right_wheel_speeds)

                updated_artists += (left_wheel_line, right_wheel_line)

            return updated_artists

        dt = t[1] - t[0]
        ani = animation.FuncAnimation(
            fig, update, frames=len(t), blit=True, interval=dt*1000)
        plt.show()

    def __animate_robot_path_3d(self, t, global_positions, global_orientations):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        position_x_lims = (
            np.min(global_positions[:, 0]) - 1, np.max(global_positions[:, 0]) + 1)
        position_y_lims = (
            np.min(global_positions[:, 1]) - 1, np.max(global_positions[:, 1]) + 1)

        # Plot settings
        ax.set_xlim(position_x_lims)
        ax.set_ylim(position_y_lims)
        ax.set_zlim(
            (np.min(global_positions[:, 2]) - 1, np.max(global_positions[:, 2]) + 1))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Path and Orientation")

        orientation_line = ax.plot([], [], [], color='red', lw=2)[0]

        lines = [ax.plot([], [], [], lw=2, alpha=i / len(t),
                         color='blue')[0] for i in range(len(t) - 1)]

        def init():
            orientation_line.set_data_3d([], [], [])
            for line in lines:
                line.set_data_3d([], [], [])
            return [orientation_line, *lines]

        def update(frame):
            # Update path
            for i, line in enumerate(lines[:frame]):
                line.set_data_3d(*global_positions[i:i + 2].T)

            # Update orientation line
            orientation_u = np.cos(global_orientations[frame])
            orientation_v = np.sin(global_orientations[frame])
            z_position = global_positions[frame, 2]
            orientation_line.set_data_3d(
                *np.column_stack((global_positions[frame], global_positions[frame] + np.array([orientation_u, orientation_v, z_position]))))

            return [orientation_line, *lines]

        ani = animation.FuncAnimation(fig, update, frames=len(
            t), init_func=init, blit=True, interval=50)
        plt.show()

    def __animate_manipulator_3d(self, t, end_effector_global_positions, end_effector_global_orientations):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(num, t, positions, orientations, line, quiver):
            ax.clear()
            ax.set_title(f"Time: {t[num]:.2f}s")

            ax.scatter(*positions[num], c='r', marker='o')
            line.set_data(positions[:num+1, 0], positions[:num+1, 1])
            line.set_3d_properties(positions[:num+1, 2])

            R = orientations[num]
            quiver = ax.quiver(
                *positions[num], *R[:, 0], color='r', length=0.2)
            ax.quiver(*positions[num], *R[:, 1], color='g', length=0.2)
            ax.quiver(*positions[num], *R[:, 2], color='b', length=0.2)

            ax.set_xlim([np.min(positions[:, 0]) - 0.5,
                        np.max(positions[:, 0]) + 0.5])
            ax.set_ylim([np.min(positions[:, 1]) - 0.5,
                        np.max(positions[:, 1]) + 0.5])
            ax.set_zlim([np.min(positions[:, 2]) - 0.5,
                        np.max(positions[:, 2]) + 0.5])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            return line, quiver

        line, = ax.plot([], [], [], lw=2)
        quiver = None

        positions = np.array(end_effector_global_positions)
        orientations = np.array(end_effector_global_orientations)

        ani = FuncAnimation(fig, update, len(t), fargs=(t, positions, orientations, line, quiver),
                            interval=50, blit=False)

        plt.show()

    def run_wheel_speed_sim(self,
                            left_wheel_speed: float,
                            right_wheel_speed: float,
                            dt: float,
                            time: float,
                            three_d: bool = False,):
        t, global_positions, global_orientations = self.forward_kin_sim(
            left_wheel_speed, right_wheel_speed, dt, time)
        if three_d:
            self.__animate_robot_path_3d(
                t, global_positions, global_orientations)
        else:
            self.__animate_robot_path_2d(
                t, global_positions, global_orientations)

    def run_path_trace_sim(self,
                           path: Path,
                           dt: float,
                           three_d: bool = False,
                           bg_img_path=None,
                           bg_img_scale=None):
        all_seg_t, all_seg_global_positions, all_seg_global_orientations, all_seg_right_wheel_speeds, all_seg_left_wheel_speeds = self.inverse_kin_sim(
            path, dt)
        t = np.concatenate(all_seg_t, axis=0)
        global_positions = np.concatenate(all_seg_global_positions, axis=0)
        global_orientations = np.concatenate(
            all_seg_global_orientations, axis=0)
        left_wheel_speeds = np.concatenate(
            all_seg_left_wheel_speeds, axis=0)
        right_wheel_speeds = np.concatenate(
            all_seg_right_wheel_speeds, axis=0)

        if three_d:
            self.__animate_robot_path_3d(
                t, global_positions, global_orientations)
        else:
            self.__animate_robot_path_2d(
                t, global_positions, global_orientations, path, bg_img_path, left_wheel_speeds, right_wheel_speeds, bg_img_scale)
        path_time = t[-1]
        final_position = global_positions[-1]
        final_orientation = global_orientations[-1]
        return path_time, final_position, final_orientation

    def run_manipulator_sim(self, n_via_points: int, total_time: float, dt: float, start_time: float = 0.0):
        target_position = self.transform_to_robot_frame(
            self.target_global_position)
        self.quad_arm.add_target(target_position)
        t, t_via_points, via_points, joint_values_at_via_points, all_thetas, all_velocities, all_accelerations, end_effector_positions, end_effector_orientations = self.quad_arm.move_to_target_sim(
            n_via_points, total_time, dt)
        end_effector_global_positions = self.transform_to_global_frame(
            end_effector_positions)
        end_effector_global_orientations = self.transform_to_global_frame(
            end_effector_orientations)
        self.__animate_manipulator_3d(
            t, end_effector_global_positions, end_effector_global_orientations)
        return t + start_time, all_thetas.T, all_velocities.T, all_accelerations.T, end_effector_global_positions, end_effector_global_orientations
