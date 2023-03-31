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


def get_angle_segments(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    v1 = [x2 - x1, y2 - y1]
    v2 = [x4 - x3, y4 - y3]
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return angle


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


def find_arc_center(p1, p2, radius):
    # Calculate the midpoint of the line segment connecting the two endpoints
    midpoint = (p1 + p2) / 2
    # Calculate the normalized vector along the line segment
    delta = p2 - p1
    chord_length = np.linalg.norm(delta)
    delta_normalized = delta / chord_length
    # Calculate the normalized vector perpendicular to the line segment
    perpendicular_normalized = np.array(
        [-delta_normalized[1], delta_normalized[0]])
    # Calculate the distance from the midpoint to the center of the arc
    distance_to_center = np.sqrt(radius**2 - (chord_length / 2)**2)
    # Find the coordinates of the center of the arc
    center1 = midpoint + distance_to_center * perpendicular_normalized
    center2 = midpoint - distance_to_center * perpendicular_normalized
    return tuple(center1), tuple(center2)


def create_fillet(line1, line2, apex_offset):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    intersection = np.array([x2, y2])
    line1_vector = np.array([x2 - x1, y2 - y1])
    line2_vector = np.array([x4 - x3, y4 - y3])
    line1_unit_vector = line1_vector / np.linalg.norm(line1_vector)
    line2_unit_vector = line2_vector / np.linalg.norm(line2_vector)
    angle_between_lines = np.arccos(
        np.dot(line1_unit_vector, line2_unit_vector))
    fillet_radius = apex_offset / (1 - np.sin(angle_between_lines / 2))
    distance_from_intersection = fillet_radius / \
        np.tan(angle_between_lines / 2)
    line1_end_point = intersection - line1_unit_vector*distance_from_intersection
    line2_start_point = intersection + line2_unit_vector*distance_from_intersection
    shortened_line1 = np.array([(x1, y1), line1_end_point])
    shortened_line2 = np.array([line2_start_point, (x4, y4)])
    fillet_centers = find_arc_center(
        line1_end_point, line2_start_point, fillet_radius)
    # find the center furthest from the intersection
    distances_from_intersection = [get_distance_between_points(
        intersection, center) for center in fillet_centers]
    fillet_center = fillet_centers[np.argmax(distances_from_intersection)]
    return shortened_line1, shortened_line2, fillet_center, fillet_radius


def distance_between_segments(s1, s2):
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


def get_arc_angles(center, point1, point2):
    arc_start_angle = np.rad2deg(np.arctan2(
        point1[1] - center[1], point1[0] - center[0]))
    arc_end_angle = np.rad2deg(np.arctan2(
        point2[1] - center[1], point2[0] - center[0]))
    return arc_start_angle, arc_end_angle
