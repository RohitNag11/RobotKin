import numpy as np


def create_t_matrix(orientation, pos):
    t_mat = np.concatenate((orientation, np.array([pos]).T), axis=1)
    t_mat = np.concatenate((t_mat, np.array([[0, 0, 0, 1]])), axis=0)
    return t_mat


def get_link_t_matrix(link_length, link_twist, link_offset, joint_angle):
    c_t = np.cos(joint_angle)
    s_t = np.sin(joint_angle)
    c_a = np.cos(link_twist)
    s_a = np.sin(link_twist)
    return np.array([[c_t, -s_t, 0, link_length],
                     [s_t*c_a, c_t*c_a, -s_a, -s_a*link_offset],
                     [s_t*s_a, c_t*s_a, c_a, c_a*link_offset],
                     [0, 0, 0, 1]])


def get_distance_between_points(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def get_pick_up_path_points(y, height, no_points):
    points = np.zeros((no_points, 3))
    points[:, 1] = y
    point_labels = np.arange(0, no_points)
    points[:, 0] = height * np.cos(np.pi * point_labels / (no_points - 1))
    points[:, 2] = height * np.sin(np.pi * point_labels / (no_points - 1))
    return points
