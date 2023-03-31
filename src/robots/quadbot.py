from src.mainpulator_base_classes import Robot
import numpy as np


class QuadBot(Robot):
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
                 d_3_range=[0, 100],
                 FLOOR_Z=0):
        super().__init__(name='hexabot',
                         origin=np.array([0, 0, 0]),
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

    def add_target(self, target):
        self.target = target
        theta_1, theta_2, d_3, theta_4 = self.inverse_kin(
            target)
        return theta_1, theta_2, d_3, theta_4

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
