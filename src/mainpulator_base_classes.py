import numpy as np
from .utils import (geometry as geom,
                    math as mth)


class Manipulator:
    def __init__(self,
                 name='arm',
                 origin=np.array([0, 0, 0]),
                 orientation=np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])):
        self.origin = origin
        self.orientation = orientation
        self.joint_link_orders = {}
        self.joint_links = []

        self.add_joint_link("base", "fixed", 0, 0, 0, 0)
        self.base = self.joint_links[0]
        self.end_effector = None
        self.chain_complete = False
        # self.joint_links[0].set_global_position(self.origin)
        # self.joint_links[0].set_global_orientation(self.orientation)

    def add_joint_link(self,
                       name,
                       joint_type,
                       link_length,
                       link_twist,
                       link_offset,
                       joint_angle,
                       joint_range=None):
        self.joint_link_orders[name] = len(self.joint_links)
        if joint_type == "fixed":
            jointLink = FixedJointLink(name,
                                       link_length,
                                       link_twist,
                                       link_offset,
                                       joint_angle)
        elif joint_type == "revolute":
            jointLink = RevoluteJointLink(name,
                                          link_length,
                                          link_twist,
                                          link_offset,
                                          joint_angle,
                                          joint_range)
        elif joint_type == "prismatic":
            jointLink = PrismaticJointLink(name,
                                           link_length,
                                           link_twist,
                                           link_offset,
                                           joint_angle,
                                           joint_range)
        self.joint_links.append(jointLink)

    @property
    def adj_t_matrices(self):
        return np.array([
            geom.get_link_t_matrix(m.link_length,
                                   m.link_twist,
                                   m.link_offset,
                                   m.joint_angle) for m in self.joint_links])

    def complete_chain(self):
        self.update_joint_link_positions()
        if self.end_effector is None:
            self.end_effector = self.get_i_th_joint_link(-1)
        self.chain_complete = True

    def update_joint_link_positions(self):
        for m in range(0, len(self.joint_links)):
            adj_t_matrices = self.adj_t_matrices[:m+1]
            origin_m_t_matrix = mth.chain_mat_mul(adj_t_matrices)
            translation_vector = origin_m_t_matrix[:3, 3]
            rotation_matrix = origin_m_t_matrix[:3, :3]
            m_global_position = self.origin + translation_vector
            m_global_orientation = np.matmul(rotation_matrix, self.orientation)
            self.joint_links[m].set_global_position(m_global_position)
            self.joint_links[m].set_global_orientation(m_global_orientation)

    def update_joint_link_angles(self, thetas):
        for i, jl in enumerate(self.joint_links[1:]):
            if jl.type == 'revolute':
                jl.set_joint_angle(thetas[i])
            elif jl.type == 'prismatic':
                jl.set_link_offset(thetas[i])
        self.update_joint_link_positions()

    def get_joint_link(self, joint_link_name=None, joint_link_index=None):
        if joint_link_name is not None:
            return self.joint_links[self.joint_link_orders[joint_link_name]]
        else:
            return self.joint_links[joint_link_index]

    def get_i_th_joint_link(self, i):
        return self.joint_links[i]


class FixedJointLink:
    def __init__(self,
                 name: str,
                 link_length: float = 0,
                 link_twist: float = 0,
                 link_offset: float = 0,
                 joint_angle: float = 0):
        self.name = name
        self.type = 'fixed'
        self.link_length = link_length
        self.link_twist = link_twist
        self.link_offset = link_offset
        self.joint_angle = joint_angle

    def set_global_position(self, global_position: np.array([0, 0, 0])):
        self.global_position = global_position

    def set_global_orientation(self, global_orientation: np.array([0, 0, 0])):
        self.global_orientation = global_orientation


class RevoluteJointLink(FixedJointLink):
    def __init__(self,
                 name: str,
                 link_length: float = 0,
                 link_twist: float = 0,
                 link_offset: float = 0,
                 joint_angle: float = 0,
                 rotation_range: tuple = (-np.pi, np.pi)):
        super().__init__(name,
                         link_length,
                         link_twist,
                         link_offset,
                         joint_angle)
        self.type = 'revolute'
        self.rotation_range = rotation_range

    def set_joint_angle(self, joint_angle: float):
        # normalise joint angle to [-pi, pi]
        joint_angle = np.arctan2(np.sin(joint_angle), np.cos(joint_angle))
        if self.rotation_range[0] <= joint_angle <= self.rotation_range[1]:
            self.joint_angle = joint_angle
        else:
            print(
                f"{self.name}'s joint angle is {joint_angle} but should be between {self.rotation_range[0]} and {self.rotation_range[1]}")


class PrismaticJointLink(FixedJointLink):
    def __init__(self,
                 name: str,
                 link_length: float = 0,
                 link_twist: float = 0,
                 link_offset: float = 0,
                 joint_angle: float = 0,
                 extension_range: tuple = (0, 50)):
        super().__init__(name,
                         link_length,
                         link_twist,
                         link_offset,
                         joint_angle)
        self.type = 'prismatic'
        self.translation_range = (extension_range[0],
                                  extension_range[1])

    def set_link_offset(self, link_offset: float):
        if self.translation_range[0] <= link_offset <= self.translation_range[1]:
            self.link_offset = link_offset
        else:
            print(
                f"{self.name}'s link offset is {link_offset} but should be between {self.translation_range[0]} and {self.translation_range[1]}")
