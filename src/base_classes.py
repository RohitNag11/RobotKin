import numpy as np
from .utils import (geometry as geom,
                    math as mth)


class Robot:
    def __init__(self,
                 name='robot',
                 origin=np.array([0, 0, 0]),
                 orientation=np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])):
        self.origin = origin
        self.orientation = orientation
        self.joint_link_orders = {}
        self.joint_links = []

        self.add_joint_link("base", "fixed", 0, 0, 0, 0)
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

    def complete_chain(self):
        self.adj_t_matrices = np.array([
            geom.get_link_t_matrix(m.link_length,
                                   m.link_twist,
                                   m.link_offset,
                                   m.joint_angle) for m in self.joint_links])
        self.update_joint_link_positions()

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


class FixedJointLink:
    def __init__(self,
                 name: str,
                 link_length: float = 0,
                 link_twist: float = 0,
                 link_offset: float = 0,
                 joint_angle: float = 0):
        self.name = name
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
        self.joint_range = rotation_range

    def set_joint_angle(self, joint_angle: float):
        if self.rotation_range[0] <= joint_angle <= self.rotation_range[1]:
            self.joint_angle = joint_angle
        else:
            raise ("Joint angle out of range")


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
        self.fixed_offset = link_offset
        self.translation_range = (extension_range[0] + self.fixed_offset,
                                  extension_range[1] + self.fixed_offset)

    def set_link_offset(self, link_offset: float):
        if self.translation_range[0] <= link_offset <= self.translation_range[1]:
            self.joint_angle = link_offset
        else:
            raise ("Link offset out of range")
