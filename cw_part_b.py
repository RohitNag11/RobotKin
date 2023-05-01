'''
Authors: Rohit Nag
Date created: 04/20/2023
Usage: 
    1. Set the type of simulation in __main__:
        1.a. 'wheel_speed_sim' for forward kinematic simulation
        1.b. 'img_path_trace_sim' for path tracing simulation with an image (inverse kinematics)
            1.b.note: set the image path in get_img_path_params()
        1.c. 'manual_path_trace_sim' for path tracing simulation with manual path setting (inverse kinematics)
            1.c.note: set the path in get_manual_input_path_params()
        'img_path_trace_manipulator_sim' for:
            - path tracing simulation with an image (inverse kinematics)
            - once the path is traced, the manipulator attempts to reach the target within the image (inverse kinematics)
    2. Set the robot parameters in get_mobile_robot_params().
    3. Set the update rate in get_update_params().
    4. Set the animation parameters in get_animation_params().
        4.note: If you want to view the end effector position of the manipulator, set the view to 3d in get_animation_params().
    5. Run the script using:
        'python3 cw_part_b.py'
'''


from src.robots import DiffDriveQuadBot, QuadArm
from src.utils.path import Path, LinearSegment, CircularSegment
from src.utils import (image as img_utils,
                       geometry as geom)
import cv2
import numpy as np


def get_mobile_robot_params():
    # INPUT! Set the robot parameters here
    # Note: changing the type of robot is not currently supported.
    # Note: global_position_init and global_orientation_init are not used when running path trace sim.
    return {'type': 'diff_drive',
            'wheel_offset': 0.8,
            'wheel_radius': 0.35,
            'max_rpm': 200,
            'global_position_init': [0, 0, 0],
            'global_orientation_init': 0,
            'width': 2,
            'chassis_height': 0.2}


def get_update_params():
    # INPUT! Set the update rate here.
    return {'dt': 0.03}


def get_animation_params():
    # INPUT! Set the animation parameters here.
    # Note: 3d plots are not supported when running path sim with an image.
    return {'three_d': False}


def get_wheel_speed_sim_params():
    # INPUT! Set wheel speed simulation (forward kinematics) parameters here.
    # Note: only change if running wheel speed sim.
    return {'left_wheel_speed': 15,
            'right_wheel_speed': 10,
            'time': 20}


def get_manual_input_path_params():
    # INPUT! Set path trace simulation (inverse kinematics) parameters here.
    # Note: only required if running path trace sim without an image.
    return [
        {'start_pos': [0, 0], 'end_pos': [1000, 0], 'type': 'linear'},
        {'start_pos': [1000, 0], 'end_pos': [1100, -100],
            'center_pos': [1000, -100], 'type': 'circular'},
        {'start_pos': [1100, -100], 'end_pos': [1100, -2000], 'type': 'linear'}
    ]


def get_image_params():
    # INPUT! Set image parameters here.
    # Note: only required if running path trace sim with an image.
    # Note: image_width_m is the width of the image in meters.
    # Note: colors are in BGR format.
    # Note: output_image_paths are the paths to save the output images. Set to None if not required.
    return {
        'image_path': 'images/CWMap.jpg',
        'road_bounds_col': (2, 44, 217),
        'start_fin_col': (251, 0, 1),
        'manipulator_target_col': (60, 255, 144),
        'image_width_m': 500,
        'output_image_paths': {
            'manipulator_target': 'images/manipulator_target_pos.jpg',
            'start_finish': 'images/start_finish_pos.jpg',
            'road_bounds': 'images/road.jpg',
            'manipulator_target': 'images/manipulator_target_pos.jpg',
        }
    }


def get_quad_manipulator_params():
    return {
        'theta_1_init': np.pi,
        'theta_2_init': -np.pi,
        'theta_4_init': -np.pi,
        'd_3_init': 0,
        'L_1': 6,
        'L_2': 6,
        'L_4': 5,
        'L_E': 2,
        'd_3_range': [0, 4]
    }


def get_manipulator_motion_params():
    return {
        'n_via_points': 10,
        'total_time': 10,
        'dt': 0.1,
    }


def get_m_to_px_ratio():
    image_path = get_image_params()['image_path']
    image_width_m = get_image_params()['image_width_m']
    raw_img = cv2.imread(image_path)
    img_dim = (raw_img.shape[0], raw_img.shape[1])
    img_scale = image_width_m / img_dim[1]
    return img_scale


def pre_process_img(image_path: str,
                    road_bounds_col: tuple,
                    start_fin_col: tuple,
                    manipulator_target_col: tuple,
                    image_width_m: float,
                    output_image_paths: dict[str, str]):
    raw_img = cv2.imread(image_path)
    img_dim = (raw_img.shape[0], raw_img.shape[1])
    img_scale = get_m_to_px_ratio()
    # Get the manipulator target image:
    target_img = img_utils.filter_colors(
        image_path, [manipulator_target_col], tolerance=30)
    target_binary_img = cv2.threshold(
        target_img, 128, 255, cv2.THRESH_BINARY)[1]
    target_gray_img = cv2.cvtColor(target_binary_img, cv2.COLOR_BGR2GRAY)
    target_gray_img = cv2.medianBlur(target_gray_img, 5)
    # Get the start and end markers image:
    start_fin_img = img_utils.filter_colors(
        image_path, [start_fin_col], tolerance=30)
    start_fin_binary_img = cv2.threshold(
        start_fin_img, 128, 255, cv2.THRESH_BINARY)[1]
    start_fin_gray_img = cv2.cvtColor(start_fin_binary_img, cv2.COLOR_BGR2GRAY)
    start_fin_gray_img = cv2.medianBlur(start_fin_gray_img, 5)
    road_bounds_img = img_utils.filter_colors(
        image_path, [road_bounds_col], tolerance=30)
    road_bounds_binary_img = cv2.threshold(
        road_bounds_img, 128, 255, cv2.THRESH_BINARY)[1]
    road_bounds_gray_img = cv2.cvtColor(
        road_bounds_binary_img, cv2.COLOR_BGR2GRAY)

    if output_image_paths['manipulator_target']:
        cv2.imwrite(output_image_paths['manipulator_target'], target_img)
    if output_image_paths['start_finish']:
        cv2.imwrite(output_image_paths['start_finish'], start_fin_img)
    if output_image_paths['road_bounds']:
        cv2.imwrite(output_image_paths['road_bounds'], road_bounds_img)
    return img_dim, raw_img, target_gray_img, start_fin_gray_img, road_bounds_gray_img


def get_manipulator_target_pos_from_img(target_gray_img):
    detected_circles = cv2.HoughCircles(target_gray_img,
                                        cv2.HOUGH_GRADIENT,
                                        dp=1,
                                        minDist=20,
                                        param1=1,
                                        param2=20,
                                        minRadius=0,
                                        maxRadius=100)
    target_px_pos = None
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        target_px_pos = detected_circles[0][0][:2]
        print(
            f'manipulator target position: ({target_px_pos[0]}, {target_px_pos[1]})')
    return [*target_px_pos, 0]


def get_start_finish_pos_from_img(start_fin_gray_img):
    detected_circles = cv2.HoughCircles(start_fin_gray_img,
                                        cv2.HOUGH_GRADIENT,
                                        dp=1,
                                        minDist=20,
                                        param1=1,
                                        param2=20,
                                        minRadius=0,
                                        maxRadius=100)
    start_px_pos = end_px_pos = None
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        end_px_pos, start_px_pos = detected_circles[0][0][:2], detected_circles[0][1][:2]
        print(f'start position: ({start_px_pos[0]}, {start_px_pos[1]})')
        print(f'end position: ({end_px_pos[0]}, {end_px_pos[1]})')
    return start_px_pos, end_px_pos


def get_road_edge_lines_from_img(road_bounds_gray_img):
    road_edge_map = cv2.Canny(road_bounds_gray_img, 50, 150)
    road_edge_lines = cv2.HoughLinesP(road_edge_map, 1, np.pi/180,
                                      threshold=60, minLineLength=100, maxLineGap=30)
    save_road_edge_lines_img(road_edge_lines, road_bounds_gray_img)
    return road_edge_lines


def save_road_edge_lines_img(road_edge_lines, road_bounds_gray_img):
    road_bounds_bgr_img = cv2.cvtColor(
        road_bounds_gray_img, cv2.COLOR_GRAY2BGR)
    for line in road_edge_lines:
        x1, y1, x2, y2 = line[0]
        road_bounds_bgr_img = cv2.line(
            road_bounds_bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 6)
    cv2.imwrite('images/road_edge_lines.png', road_bounds_bgr_img)


def get_path_from_road_edge_bounds(start: tuple,
                                   end: tuple,
                                   road_edge_lines: list,
                                   img_width: int,
                                   img_height: int,
                                   robot_width_px: float):
    start_x, start_y = start
    end_x, end_y = end
    max_dist1 = max_dist2 = 0
    end_point1 = start_point2 = None
    for dummy_line in road_edge_lines:
        slope, _ = geom.get_equation_of_line(dummy_line[0])
        x12 = start_x - start_y / slope
        x22 = end_x - end_y / slope
        line1 = (start_x, start_y, x12, 0)
        line2 = (end_x, end_y, x22, 0)
        for line in road_edge_lines:
            line_start_x, line_start_y, line_end_x, line_end_y = line[0]
            slope3 = (line_end_y - line_start_y) / (line_end_x - line_start_x)
            if slope3 != slope:
                intersection1 = geom.get_line_segment_intersect(line1, line[0])
                intersection2 = geom.get_line_segment_intersect(line2, line[0])
                if intersection1 and 0 <= intersection1[0] <= img_width and 0 <= intersection1[1] <= img_height:
                    dist1 = geom.get_distance_between_points(
                        (start_x, start_y), intersection1)
                    if dist1 > max_dist1:
                        max_dist1 = dist1
                        end_point1 = intersection1
                if intersection2 and 0 <= intersection2[0] <= img_width and 0 <= intersection2[1] <= img_height:
                    dist2 = geom.get_distance_between_points(
                        (end_x, end_y), intersection2)
                    if dist2 > max_dist2:
                        max_dist2 = dist2
                        start_point2 = intersection2
    if end_point1 and start_point2:
        line1 = (start_x, start_y, end_point1[0], end_point1[1])
        line2 = (end_x, end_y, start_point2[0], start_point2[1])
        intersection = geom.get_line_segment_intersect(line1, line2)
        line1 = (start_x, start_y, intersection[0], intersection[1])
        line2 = (intersection[0], intersection[1], end_x, end_y)
        min_dist_to_road = min([geom.get_distance_between_segments(line, s[0])
                                for s in road_edge_lines for line in [line1, line2]])
        l1 = np.array([[line1[0], line1[1]], [line1[2], line1[3]]])
        l2 = np.array([[line2[0], line2[1]], [line2[2], line2[3]]])
        d_theta = geom.get_internal_angle_between_lines(l1, l2)
        external_distance = min_dist_to_road / \
            np.sin(d_theta / 2) - 1.1 * robot_width_px / 2
        shortened_line1, shortened_line2, curvature_center, curvature_radius = geom.fillet_segments(
            l1, l2, external_distance)
        segment_1 = LinearSegment(shortened_line1[0],
                                  shortened_line1[1])
        segment_2 = CircularSegment(shortened_line1[1],
                                    shortened_line2[0],
                                    curvature_center)
        segment_3 = LinearSegment(shortened_line2[0],
                                  shortened_line2[1])
        return Path([segment_1, segment_2, segment_3])
    else:
        return None


def get_path_from_manual_input():
    segments = []
    for segment in get_manual_input_path_params():
        seg_type = segment.pop('type')
        if seg_type == 'linear':
            segments.append(LinearSegment(**segment))
        elif seg_type == 'circular':
            segments.append(CircularSegment(**segment))
    return Path(segments=segments)


def get_path_and_target_from_img():
    img_scale = get_m_to_px_ratio()
    img_dim, raw_img, target_gray_img, start_fin_gray_img, road_bounds_gray_img = pre_process_img(
        **get_image_params())
    img_height, img_width = img_dim
    manip_target_pos = get_manipulator_target_pos_from_img(target_gray_img)
    start_pos, end_pos = get_start_finish_pos_from_img(start_fin_gray_img)
    road_edge_lines = get_road_edge_lines_from_img(road_bounds_gray_img)
    robot_width_px = get_mobile_robot_params()['width'] / img_scale
    path = get_path_from_road_edge_bounds(start_pos,
                                          end_pos,
                                          road_edge_lines,
                                          img_width,
                                          img_height,
                                          robot_width_px)
    return path, manip_target_pos


def run_wheel_speed_sim(mobile_bot):
    mobile_bot.run_wheel_speed_sim(**get_wheel_speed_sim_params(),
                                   **get_update_params(),
                                   **get_animation_params())


def run_path_trace_sim(mobile_bot, path, img_path=None, img_scale=None):
    mobile_bot.run_path_trace_sim(path=path,
                                  bg_img_path=img_path,
                                  bg_img_scale=img_scale,
                                  **get_update_params(),
                                  **get_animation_params())


def run_path_trace_sim_with_manipulator(mobile_bot, path, manip_target_pos, img_path, img_scale):
    path_time, final_position, final_orientation = mobile_bot.run_path_trace_sim(
        path=path, bg_img_path=img_path, bg_img_scale=img_scale, **get_update_params(), **get_animation_params())
    quad_arm = QuadArm(**get_quad_manipulator_params())
    mobile_bot.add_quad_arm(quad_arm)
    print(final_position)
    print(manip_target_pos)
    mobile_bot.add_arm_target(manip_target_pos)
    mobile_bot.run_manipulator_sim(
        **get_manipulator_motion_params(), start_time=path_time)


def main(demo_name='wheel_speed_sim'):
    m_px_ratio = get_m_to_px_ratio()
    robot_params = get_mobile_robot_params()
    robot_type = robot_params.pop('type')
    if robot_type != 'diff_drive':
        raise NotImplementedError(
            f"Robot type {robot_type} not supported. Please use a 'diff_drive' robot.")
    else:
        if demo_name == 'wheel_speed_sim':
            mobile_bot = DiffDriveQuadBot(**robot_params)
            run_wheel_speed_sim(mobile_bot)
        elif demo_name == 'manual_path_trace_sim':
            mobile_bot = DiffDriveQuadBot(**robot_params)
            run_path_trace_sim(mobile_bot, get_path_from_manual_input())
        elif demo_name == 'img_path_trace_sim' or demo_name == 'img_path_trace_manipulator_sim':
            robot_params_to_scale = [
                'width', 'wheel_radius', 'wheel_offset', 'chassis_height']
            [robot_params.update({param: robot_params[param] / m_px_ratio})
             for param in robot_params_to_scale]
            mobile_bot = DiffDriveQuadBot(**robot_params)
            path, manip_target_pos = get_path_and_target_from_img()
            img_path = get_image_params()['image_path']
            img = cv2.imread(img_path)
            img_x_dim = img.shape[1]
            img_y_dim = img.shape[0]
            if demo_name == 'img_path_trace_sim':
                run_path_trace_sim(mobile_bot, path, img_path, m_px_ratio)
            else:
                run_path_trace_sim_with_manipulator(
                    mobile_bot, path, manip_target_pos, img_path, m_px_ratio)


if __name__ == '__main__':
    # 'wheel_speed_sim' or 'img_path_trace_sim' or 'manual_path_trace_sim' or 'img_path_trace_manipulator_sim'
    demo_name = 'img_path_trace_manipulator_sim'
    main(demo_name)
