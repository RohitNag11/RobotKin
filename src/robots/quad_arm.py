from src.mainpulator_base_classes import Manipulator
import numpy as np
from src.utils import geometry as geom


class QuadArm(Manipulator):
    def __init__(self,
                 theta_1_init=0,
                 theta_2_init=0,
                 theta_4_init=0,
                 d_3_init=0,
                 L_1=60,
                 L_2=40,
                 L_4=10,
                 L_E=30,
                 theta_1_range=[-np.pi/2, np.pi/2],
                 theta_2_range=[-np.pi/2, np.pi/2],
                 theta_4_range=[-np.pi/2, np.pi/2],
                 d_3_range=[-100, 100],
                 FLOOR_Z=0,
                 origin=np.array([0, 0, 0])):
        super().__init__(name='quadbot',
                         origin=origin,
                         orientation=np.array([[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]]))
        self.d_3 = d_3_init
        self.theta_1 = theta_1_init
        self.theta_2 = theta_2_init
        self.theta_4 = theta_4_init
        self.L_1 = L_1
        self.L_2 = L_2
        self.L_4 = L_4
        self.L_E = L_E
        self.theta_1_range = theta_1_range
        self.theta_2_range = theta_2_range
        self.theta_4_range = theta_4_range
        self.d_3_range = d_3_range
        self.FLOOR_Z = FLOOR_Z

        self.__initialise_robot()

    def __initialise_robot(self):
        self.add_joint_link(name='1_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=0.0,
                            link_offset=self.L_1,
                            joint_angle=self.theta_1,
                            joint_range=self.theta_1_range)
        self.add_joint_link(name='2_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=-np.pi/2,
                            link_offset=self.L_2,
                            joint_angle=self.theta_2,
                            joint_range=self.theta_2_range)
        self.add_joint_link(name='3_prismatic',
                            joint_type='prismatic',
                            link_length=0.0,
                            link_twist=-np.pi/2,
                            link_offset=self.d_3,
                            joint_angle=0,
                            joint_range=self.d_3_range)
        self.add_joint_link(name='4_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=np.pi/2,
                            link_offset=0.0,
                            joint_angle=self.theta_4,
                            joint_range=self.theta_4_range)
        self.add_joint_link(name='end_effector',
                            joint_type='fixed',
                            link_length=self.L_E,
                            link_twist=0.0,
                            link_offset=self.L_4,
                            joint_angle=0.0)
        self.complete_chain()

    def get_end_effector_position(self):
        return self.forward_kin_pos([self.theta_1,
                                     self.theta_2,
                                     self.d_3,
                                     self.theta_4])

    def get_work_space(self, no_points=100):
        q_ranges = np.array([self.theta_1_range, self.theta_2_range,
                             self.d_3_range, self.theta_4_range])
        q_min, q_max = q_ranges[:, 0], q_ranges[:, 1]
        q_meshgrid = np.meshgrid(*[np.linspace(q_min[i], q_max[i], no_points, endpoint=True)
                                 for i in range(len(q_min))], indexing='ij')
        q_grid = np.stack(q_meshgrid, axis=-1)

        # Compute the end effector position for each joint angle
        q_flat = q_grid.reshape((-1, q_grid.shape[-1]))
        xyz_flat = np.apply_along_axis(self.forward_kin_pos, 1, q_flat)
        x_vals, y_vals, z_vals = xyz_flat.T
        # Cap the z values to the floor height
        z_vals = np.clip(z_vals, self.FLOOR_Z, None)
        return np.array([x_vals, y_vals, z_vals])

    def add_target(self, target_pos):
        self.target_position = np.array(target_pos)
        self.target_orientation = self.orientation

    def __get_linear_manipulator_trajectory_to_target(self, n_via_points=10):
        current_end_effector_local_pos = self.end_effector.global_position
        ee_x, ee_y, ee_z = current_end_effector_local_pos
        t_x, t_y, t_z = self.target_position
        vp_x = np.linspace(ee_x, t_x, n_via_points, endpoint=True)
        vp_y = np.linspace(ee_y, t_y, n_via_points, endpoint=True)
        vp_z = np.linspace(ee_z, t_z, n_via_points, endpoint=True)
        via_points = np.array([vp_x, vp_y, vp_z]).T
        return via_points

    def __get_joint_dynamics_for_motion(self, joint_values_at_via_points, total_time, t_step):
        """
        Calculate joint trajectories for given joint values and time steps.

        Args:
            total_time (float): Total time of the trajectory.
            all_joint_values (list): List of joint values at each via_point.
            t_step (float): Time step for parabolic blending.

        Returns:
            tuple: A tuple containing:
                - t (numpy.ndarray): Array of time values.
                - all_thetas (numpy.ndarray): Array of joint positions at each time step.
                - all_velocities (numpy.ndarray): Array of joint velocities at each time step.
                - all_accelerations (numpy.ndarray): Array of joint accelerations at each time step.
        """
        no_points = len(joint_values_at_via_points[0])
        # Get parabolic blend times
        t_via_points, t_arrs, t_h_arr, t_h = self.__get_parabolic_blend_times(
            total_time, no_points, t_step)
        all_thetas = []
        all_velocities = []
        all_accelerations = []
        # Compute joint trajectories for each joint
        for joint_positions in joint_values_at_via_points:
            theta_i, v_i, a_i = self.__get_joint_dynamics_at_via_point(
                t_via_points, t_arrs, t_h_arr, joint_positions)
            all_thetas.append(np.column_stack((t_h.ravel(), theta_i.ravel())))
            all_velocities.append(np.column_stack((t_h.ravel(), v_i.ravel())))
            all_accelerations.append(
                np.column_stack((t_h.ravel(), a_i.ravel())))
        t = all_thetas[0][:, 0]
        all_thetas = np.array([thetas[:, 1] for thetas in all_thetas])
        all_accelerations = np.array([accs[:, 1]
                                     for accs in all_accelerations])
        all_velocities = np.array([vels[:, 1] for vels in all_velocities])
        return t, t_via_points, all_thetas, all_velocities, all_accelerations

    def __get_parabolic_blend_times(self, total_time, n_via_points, t_step):
        """
        Calculate parabolic blend times for given total time, number of points and time step.

        Args:
            total_time (float): Total time of the trajectory.
            no_points (int): Number of points in the trajectory.
            t_step (float): Time step for parabolic blending.

        Returns:
            tuple: A tuple containing:
                - t (numpy.ndarray): Array of time values.
                - t_arrs (numpy.ndarray): Array of time step arrays.
                - t_h_arr (numpy.ndarray): Array of half time intervals.
                - t_h (numpy.ndarray): Array of time values with added time steps.
        """
        t_via_points = np.linspace(0, total_time, n_via_points)
        t_arrs = np.diff(t_via_points)[:, None] * np.arange(0, 1, t_step)
        t_h_arr = np.diff(t_via_points)[:, None] / 2
        t_h = t_via_points[:-1, None] + t_arrs
        return t_via_points, t_arrs, t_h_arr, t_h

    def __get_joint_dynamics_at_via_point(self, t_via_points, t_arrs, t_h_arr, joint_positions):
        """
        Compute joint dynamics at a given via point's joint positions using parabolic blending.

        Args:
            t (numpy.ndarray): Array of time values.
            t_arrs (numpy.ndarray): Array of time step arrays.
            t_h_arr (numpy.ndarray): Array of half time intervals.
            joint_positions (numpy.ndarray): Array of joint positions.å

        Returns:
            tuple: A tuple containing:
                - theta_i (numpy.ndarray): Array of joint positions at each time step.
                - v_i (numpy.ndarray): Array of joint velocities at each time step.
                - a_i (numpy.ndarray): Array of joint accelerations at each time step.
        """
        # Calculate initial and final joint positions, and the position difference
        theta_0_arr = joint_positions[:-1, None]
        theta_f_arr = joint_positions[1:, None]
        d_theta_arr = np.diff(joint_positions)[:, None]
        # Calculate acceleration and velocity arrays
        a_arr = d_theta_arr / t_h_arr**2
        v_arr = a_arr * t_h_arr
        # Create a mask to determine parabolic blending
        mask = t_arrs <= t_h_arr
        # Calculate joint positions, velocities, and accelerations based on the mask
        theta_i = np.where(mask, theta_0_arr + (v_arr / np.diff(t_via_points)[:, None]) * t_arrs**2,
                           theta_f_arr - (a_arr * np.diff(t_via_points)[:, None]**2) / 2 + a_arr * np.diff(t_via_points)[:, None] * t_arrs - (a_arr / 2) * t_arrs**2)
        v_i = np.where(mask, t_arrs * v_arr / t_h_arr, a_arr *
                       np.diff(t_via_points)[:, None] - a_arr * t_arrs**2)
        a_i = np.where(mask, v_arr / t_h_arr, -2 * a_arr * t_arrs**2)
        return theta_i, v_i, a_i

    def move_to_target_in_joint_space(self, n_via_points=10, total_time=10, t_step=0.01):
        via_points = self.__get_linear_manipulator_trajectory_to_target(
            n_via_points)
        target_orientation = self.target_orientation
        vp_t_matrices = [geom.create_t_matrix(target_orientation, vp)
                         for vp in via_points]
        joint_values_at_via_points = np.array([self.inverse_kin(target)
                                               for target in vp_t_matrices]).T
        t, t_via_points, all_thetas, all_velocities, all_accelerations = self.__get_joint_dynamics_for_motion(
            joint_values_at_via_points, total_time, t_step)
        for thetas in all_thetas.T:
            self.update_joint_link_angles(thetas)
        return t, t_via_points, all_thetas, all_velocities, all_accelerations, via_points, joint_values_at_via_points

    def traverse_via_points_in_joint_space(self, via_points, target_orientation, total_time=10, t_step=0.01):
        vp_t_matrices = [geom.create_t_matrix(target_orientation, vp)
                         for vp in via_points]
        joint_values_at_via_points = np.array([self.inverse_kin(target)
                                               for target in vp_t_matrices]).T
        t, t_via_points, all_thetas, all_velocities, all_accelerations = self.__get_joint_dynamics_for_motion(
            joint_values_at_via_points, total_time, t_step)
        for thetas in all_thetas.T:
            self.update_joint_link_angles(thetas)
        return t, t_via_points, all_thetas, all_velocities, all_accelerations, joint_values_at_via_points

    def move_to_target_sim(self, n_via_points=10, total_time=10, t_step=0.01):
        t, t_via_points, all_thetas, all_velocities, all_accelerations, via_points, joint_values_at_via_points = self.move_to_target_in_joint_space(
            n_via_points, total_time, t_step)
        end_effector_orientations = np.array(
            [self.forward_kin_orientation(theta) for theta in all_thetas.T])
        end_effector_positions = np.array(
            [self.forward_kin_pos(theta) for theta in all_thetas.T])
        return t, t_via_points, via_points, joint_values_at_via_points, all_thetas, all_velocities, all_accelerations, end_effector_positions, end_effector_orientations

    def inverse_kin(self, target):
        pos = target[:3, 3]
        orientation = target[:3, :3]
        X, Y, Z = pos
        r11, r12, r13 = orientation[0, :]
        r21, r22, r23 = orientation[1, :]
        r31, r32, r33 = orientation[2, :]
        theta_1 = np.arctan2(-r13, r23)
        if theta_1 != 0:
            theta_2 = np.arctan2(Y - self.L_E * r21 - r23 * (self.L_4 + self.L_2),
                                 r13 * (self.L_E + self.L_1 - Z))
        elif theta_1 != np.pi:
            theta_2 = np.arctan2(self.L_E * r11 + r13 * (self.L_4 + self.L_2) - X,
                                 r23 * (self.L_E * r31 + self.L_1 - Z))
        if theta_1 != np.pi and theta_2 != 0:
            d_3 = (self.L_E * r11 + r13 * (self.L_4 + self.L_2) - X) / \
                (np.cos(theta_1) * np.sin(theta_2))
        elif theta_2 != np.pi:
            d_3 = (self.L_E * r31 + self.L_1 - Z) / (np.cos(theta_2))
        theta_4 = np.arctan2(-r31, -r32) - theta_2
        return np.array([theta_1, theta_2, d_3, theta_4])

    def forward_kin_orientation(self, joint_values):
        theta_1, theta_2, d_3, theta_4 = joint_values
        c1 = np.cos(theta_1)
        s1 = np.sin(theta_1)
        c24 = np.cos(theta_2 + theta_4)
        s24 = np.sin(theta_2 + theta_4)
        return np.array([[c24 * c1, -s24 * c1, -s1],
                         [c24 * s1, -s24 * s1, c1],
                         [-s24, -c24, 0]])

    def forward_kin_pos(self, joint_values):
        theta_1, theta_2, d_3, theta_4 = joint_values
        c1 = np.cos(theta_1)
        s1 = np.sin(theta_1)
        c2 = np.cos(theta_2)
        s2 = np.sin(theta_2)
        c24 = np.cos(theta_2 + theta_4)
        s24 = np.sin(theta_2 + theta_4)
        x = self.L_E * c24 * c1 - self.L_4 * s1 - d_3 * c1 * s2 - self.L_2 * s1
        y = self.L_E * c24 * s1 + self.L_4 * c1 - d_3 * s1 * s2 + self.L_2 * c1
        z = -self.L_E * s24 + self.L_1 - (d_3 * c2)
        return np.array([x, y, z])
