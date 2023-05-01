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
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    return np.linalg.norm(point_1 - point_2)


def get_pick_up_path_points(y, height, no_points):
    points = np.zeros((no_points, 3))
    points[:, 1] = y
    point_labels = np.arange(0, no_points)
    points[:, 0] = height * np.cos(np.pi * point_labels / (no_points - 1))
    points[:, 2] = height * np.sin(np.pi * point_labels / (no_points - 1))
    return points


def get_equation_of_line(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


def get_extended_line_intersect(line1, line2):
    m1, c1 = get_equation_of_line(line1)
    m2, c2 = get_equation_of_line(line2)
    if m1 == m2:
        return None
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return (x, y)


def get_line_segment_intersect(line1, line2):
    # Get the intersection point of two lines where each line is defined by two points
    x_intersect, y_intersect = get_extended_line_intersect(line1, line2)
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # Check if the point of intersection lies within both line segments
    if (min(x1, x2) <= x_intersect <= max(x1, x2) and
        min(y1, y2) <= y_intersect <= max(y1, y2) and
        min(x3, x4) <= x_intersect <= max(x3, x4) and
            min(y3, y4) <= y_intersect <= max(y3, y4)):
        return (x_intersect, y_intersect)
    else:
        return None


def get_angle_between_points(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def get_distance_between_segments(s1, s2):
    def dist_point_to_line(p, a, b):
        return np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

    def dist_point_to_segment(p, a, b):
        if np.dot(b - a, p - a) < 0:
            return np.linalg.norm(p - a)
        if np.dot(a - b, p - b) < 0:
            return np.linalg.norm(p - b)
        return dist_point_to_line(p, a, b)
    A, B = np.array(s1)[:2], np.array(s1)[2:]
    C, D = np.array(s2)[:2], np.array(s2)[2:]
    return min(
        dist_point_to_segment(A, C, D),
        dist_point_to_segment(B, C, D),
        dist_point_to_segment(C, A, B),
        dist_point_to_segment(D, A, B),
    )


def get_angle_between_lines(line1, line2):
    l1 = np.array(line1)
    l2 = np.array(line2)
    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]
    # Calculate the dot product of the two lines
    dot_product = np.dot(v1, v2)
    # Calculate the magnitudes of the two lines
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    # Calculate the cosine of the angle between the lines
    cos_angle = dot_product / (mag_v1 * mag_v2)
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    return angle_rad


def get_internal_angle_between_lines(line1, line2):
    angle_rad = get_angle_between_lines(line1, line2)
    if angle_rad > np.pi:
        angle_rad = 2*np.pi - angle_rad
    return angle_rad


def fillet_segments(line1, line2, external_distance):
    l1 = np.array(line1)
    l2 = np.array(line2)
    d_theta = get_internal_angle_between_lines(l1, l2)
    c0 = external_distance
    r = c0 * np.sin(d_theta / 2) / (1 - np.sin(d_theta / 2))
    cut_length = r / np.tan(d_theta / 2)
    v1_hat = (l1[1] - l1[0]) / np.linalg.norm(l1[1] - l1[0])
    v2_hat = (l2[1] - l2[0]) / np.linalg.norm(l2[1] - l2[0])
    v1_perp = np.array([-v1_hat[1], v1_hat[0]])
    v2_perp = np.array([-v2_hat[1], v2_hat[0]])
    cut1 = l1[1] - v1_hat * cut_length
    cut2 = l2[0] + v2_hat * cut_length
    r_dummy = np.cross(cut2 - cut1, v2_perp) / np.cross(v1_perp, v2_perp)
    center = cut1 + r_dummy * v1_perp
    l1_cut = np.array([l1[0], cut1])
    l2_cut = np.array([cut2, l2[1]])
    return l1_cut, l2_cut, center, r


def get_arc_angles(start_pos, end_pos, center):
    # Convert input arrays to numpy arrays
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    center = np.array(center)

    # Calculate vectors from center to start and end positions
    start_vec = start_pos - center
    end_vec = end_pos - center

    start_vec_unit = start_vec / np.linalg.norm(start_vec)
    end_vec_unit = end_vec / np.linalg.norm(end_vec)

    clockwise = np.cross(start_vec, end_vec) < 0

    multiplier = -1 if clockwise else 1

    d_angle = np.abs(
        np.arccos(np.dot(start_vec_unit, end_vec_unit))) * multiplier
    d_angle = np.sign(d_angle) * (np.abs(d_angle) % (2 * np.pi))

    # Calcualte angle between unit vectors and [0, 1]:
    start_angle = np.arccos(
        np.dot(start_vec_unit, np.array([0, 1]))) * multiplier
    start_angle = np.sign(start_angle) * (np.abs(start_angle) % (2 * np.pi))

    end_angle = d_angle + start_angle
    end_angle = np.sign(end_angle) * (np.abs(end_angle) % (2 * np.pi))

    return start_angle, end_angle, d_angle, clockwise
