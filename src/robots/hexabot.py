from src.mainpulator_base_classes import Robot
import numpy as np


# define a class that inherits from the Robot class in the utils package:
class HexaBot(Robot):
    def __init__(self,
                 theta_1_init=0,
                 theta_2_init=0,
                 theta_3_init=0,
                 theta_5_init=0,
                 theta_6_init=0,
                 d_4_init=0,
                 L_0=61.25,
                 L_2=100.0,
                 L_3=102.0,
                 L_5=55.0,
                 L_6=20.0,
                 L_E=30.0):
        super().__init__(name='hexabot',
                         origin=np.array([0, 0, 0]),
                         orientation=np.array([[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]]))
        self.d_4 = d_4_init
        self.theta_1 = theta_1_init
        self.theta_2 = theta_2_init
        self.theta_3 = theta_3_init
        self.theta_5 = theta_5_init
        self.theta_6 = theta_6_init
        self.L_0 = L_0
        self.L_2 = L_2
        self.L_3 = L_3
        self.L_5 = L_5
        self.L_6 = L_6
        self.L_E = L_E
        self.__initialise_robot()

    def __initialise_robot(self):
        self.add_joint_link(name='1_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=0.0,
                            link_offset=self.L_0,
                            joint_angle=self.theta_1)
        self.add_joint_link(name='2_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=-np.pi/2,
                            link_offset=0.0,
                            joint_angle=self.theta_2,
                            joint_range=[-np.pi/2, np.pi/2])
        self.add_joint_link(name='3_revolute',
                            joint_type='revolute',
                            link_length=self.L_2,
                            link_twist=0.0,
                            link_offset=0.0,
                            joint_angle=self.theta_3,
                            joint_range=[-np.pi, np.pi])
        self.add_joint_link(name='4_prismatic',
                            joint_type='prismatic',
                            link_length=0.0,
                            link_twist=np.pi/2,
                            link_offset=self.L_3+self.d_4,
                            joint_angle=0.0,
                            joint_range=[-np.pi/2, np.pi/2])
        self.add_joint_link(name='5_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=-np.pi/2,
                            link_offset=0.0,
                            joint_angle=self.theta_5,
                            joint_range=[-np.pi/2, np.pi/2])
        self.add_joint_link(name='6_revolute',
                            joint_type='revolute',
                            link_length=0.0,
                            link_twist=np.pi/2,
                            link_offset=self.L_5,
                            joint_angle=self.theta_6,
                            joint_range=[-np.pi/2, np.pi/2])
        self.add_joint_link(name='end_effector',
                            joint_type='fixed',
                            link_length=self.L_6,
                            link_twist=0.0,
                            link_offset=self.L_E,
                            joint_angle=0.0)
        self.complete_chain()

    def calculate_j5_position(self):
        c1 = np.cos(self.theta_1)
        s1 = np.sin(self.theta_1)
        c2 = np.cos(self.theta_2)
        s2 = np.sin(self.theta_2)
        c3 = np.cos(self.theta_3)
        s3 = np.sin(self.theta_3)
        L0 = self.L_0
        L2 = self.L_2
        L3 = self.L_3
        d4 = self.d_4
        x = (L3 + d4)*(s2*c1*c3 + s3*c1*c2) \
            + L2*c1*c2
        y = (L3 + d4)*(s1*s2*c3 + s1*s3*c2) \
            + L2*s1*c2
        z = (L3 + d4)*(c2*c3 - s2*s3) \
            - L2*s2 \
            + L0
        return x, y, z

    def calculate_j5_orientation(self):
        c1 = np.cos(self.theta_1)
        s1 = np.sin(self.theta_1)
        c2 = np.cos(self.theta_2)
        s2 = np.sin(self.theta_2)
        c3 = np.cos(self.theta_3)
        s3 = np.sin(self.theta_3)
        c5 = np.cos(self.theta_5)
        s5 = np.sin(self.theta_5)
        c6 = np.cos(self.theta_6)
        s6 = np.sin(self.theta_6)
        r11 = c1*c2*c3*c5 - s2*s3*c1*c5 - s2*c1*c3*s5 - s3*c1*c2*s5
        r12 = -c1*c2*c3*s5 + s2*s3*c1*s5 - s2*c1*c3*c5 - s3*c1*c2*c5
        r13 = -s1
        r21 = s1*c2*c3*c5 - s1*s2*s3*c5 - s1*s2*c3*s5 - s1*s3*c2*s5
        r22 = -s1*c2*c3*s5 + s1*s2*s3*s5 - s1*s2*c3*c5 - s1*s3*c2*c5
        r23 = c1
        r31 = -s2*c3*c5 - s3*c2*c5 + s2*s3*s5 - c2*c3*s5
        r32 = s2*c3*s5 + s3*c2*s5 - c2*c3*c5 + s2*s3*c5
        r33 = 0
        return np.array([[r11, r12, r13],
                         [r21, r22, r23],
                         [r31, r32, r33]])

    def calculate_j6_position(self):
        c1 = np.cos(self.theta_1)
        s1 = np.sin(self.theta_1)
        c2 = np.cos(self.theta_2)
        s2 = np.sin(self.theta_2)
        t_23 = self.theta_2 + self.theta_3
        c23 = np.cos(t_23)
        s23 = np.sin(t_23)
        t_235 = self.theta_2 + self.theta_3 + self.theta_5
        c235 = np.cos(t_235)
        s235 = np.sin(t_235)
        L0 = self.L_0
        L2 = self.L_2
        L3 = self.L_3
        d4 = self.d_4
        L5 = self.L_5
        x = c1 * (L5 * s235 + (L3 + d4) * s23 + L2*c2)
        y = s1 * (L5 * s235 + (L3 + d4) * s23 + L2*c2)
        z = L5 * c235 + (L3 + d4) * c23 - L2*s2 + L0
        return x, y, z

    def calculate_j6_orientation(self):
        c1 = np.cos(self.theta_1)
        s1 = np.sin(self.theta_1)
        c6 = np.cos(self.theta_6)
        s6 = np.sin(self.theta_6)
        t_235 = self.theta_2 + self.theta_3 + self.theta_5
        c235 = np.cos(t_235)
        s235 = np.sin(t_235)
        r11 = c1 * c6 * c235 - s1 * s6
        r12 = -s1 * c6 - c1 * s6 * c235
        r13 = c1 * s235
        r21 = s1 * c6 * c235 + c1 * s6
        r22 = c1 * c6 - s1 * s6 * c235
        r23 = s1 * s235
        r31 = -c6 * s235
        r32 = s6 * s235
        r33 = c235
        return np.array([[r11, r12, r13],
                         [r21, r22, r23],
                         [r31, r32, r33]])

    def calculate_end_effector_position(self):
        c1 = np.cos(self.theta_1)
        s1 = np.sin(self.theta_1)
        c2 = np.cos(self.theta_2)
        s2 = np.sin(self.theta_2)
        c6 = np.cos(self.theta_6)
        s6 = np.sin(self.theta_6)
        t_23 = self.theta_2 + self.theta_3
        c23 = np.cos(t_23)
        s23 = np.sin(t_23)
        t_235 = self.theta_2 + self.theta_3 + self.theta_5
        c235 = np.cos(t_235)
        s235 = np.sin(t_235)
        L0 = self.L_0
        L2 = self.L_2
        L3 = self.L_3
        d4 = self.d_4
        L5 = self.L_5
        L6 = self.L_6
        LE = self.L_E
        x = c1 * (c6 * c235 * L6 + s235 * (LE + L5) +
                  s23 * (L3 + d4) + c2 * L2) - s1 * s6 * L6
        y = s1 * (c6 * c235 * L6 + s235 * (LE + L5) +
                  s23 * (L3 + d4) + c2 * L2) + c1 * s6 * L6
        z = -c6 * s235 * L6 + c235 * (LE + L5) + c23 * (L3 + d4) - s2 * L2 + L0
        return x, y, z

    def calculate_end_effector_orientation(self):
        return self.calculate_j6_orientation()
