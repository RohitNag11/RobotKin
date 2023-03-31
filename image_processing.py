from src.utils import (image as img_utils,
                       geometry as geom,)
import cv2
import numpy as np


def get_path(start: tuple,
             end: tuple,
             surrounded_lines: list,
             img_width: int,
             img_height: int,
             car_width):
    start_x, start_y = start
    end_x, end_y = end
    max_dist1 = max_dist2 = 0
    end_point1 = start_point2 = None

    for dummy_line in surrounded_lines:
        slope, _ = geom.get_equation_of_line(dummy_line[0])

        x12 = start_x - start_y / slope
        x22 = end_x - end_y / slope
        line1 = (start_x, start_y, x12, 0)
        line2 = (end_x, end_y, x22, 0)

        for line in surrounded_lines:
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

        min_dist = min([geom.distance_between_segments(line, s[0])
                       for s in surrounded_lines for line in [line1, line2]])

        appex_offset = min_dist - car_width / 2
        shortened_line1, shortened_line2, curvature_center, curvature_radius = geom.create_fillet(
            line1, line2, appex_offset)
        path = [shortened_line1, shortened_line2]

        return path, curvature_center, curvature_radius
    else:
        return None, None, None


image_path = 'images/CWMap.jpg'
red_color = (2, 44, 217)
blue_color = (251, 0, 1)
green_color = (60, 255, 144)
raw_img = cv2.imread(image_path)
img_width, img_height = raw_img.shape[1], raw_img.shape[0]
car_width = 40

# Get position filtered image:
markers_img = img_utils.filter_colors(image_path, [blue_color], tolerance=30)
# Save the result image
cv2.imwrite('images/positions_image.jpg', markers_img)

# Get the start and end markers image:
markers_binary_img = cv2.threshold(markers_img, 128, 255, cv2.THRESH_BINARY)[1]
markers_gray_img = cv2.cvtColor(markers_binary_img, cv2.COLOR_BGR2GRAY)
markers_gray_img = cv2.medianBlur(markers_gray_img, 5)

# Get the road boundaries image:
road_img = img_utils.filter_colors(image_path, [red_color], tolerance=30)
road_binary_img = cv2.threshold(road_img, 128, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('images/path_binary_image.jpg', road_binary_img)
road_gray_img = cv2.cvtColor(road_binary_img, cv2.COLOR_BGR2GRAY)

# Detect circles
start_end_circles = cv2.HoughCircles(markers_gray_img,
                                     cv2.HOUGH_GRADIENT,
                                     dp=1,
                                     minDist=20,
                                     param1=1,
                                     param2=20,
                                     minRadius=0,
                                     maxRadius=100)
start_pos = end_pos = None
if start_end_circles is not None:
    start_end_circles = np.uint16(np.around(start_end_circles))
    end_pos, start_pos = start_end_circles[0][0][:2], start_end_circles[0][1][:2]
    print(f'start: ({start_pos[0]}, {start_pos[1]})')
    print(f'end: ({end_pos[0]}, {end_pos[1]})')

# Detect road boundaries
road_edge_map = cv2.Canny(road_gray_img, 50, 150)
road_edge_lines = cv2.HoughLinesP(road_edge_map, 1, np.pi/180,
                                  threshold=60, minLineLength=100, maxLineGap=30)

# Get the path
path_lines, path_curve_center, path_curve_radius = get_path(start_pos,
                                                            end_pos,
                                                            road_edge_lines,
                                                            img_width,
                                                            img_height,
                                                            car_width)

# Plot the path
path_overlay_img = raw_img.copy()
for line in path_lines:
    (x1, y1), (x2, y2) = line
    cv2.line(path_overlay_img, (int(x1), int(y1)),
             (int(x2), int(y2)), (0, 255, 0), 6)
img_utils.plot_arc(path_overlay_img,
                   path_lines[0][1],
                   path_lines[1][0],
                   path_curve_center,
                   path_curve_radius,
                   (0, 255, 0),
                   6)
cv2.imwrite('images/path_overlay_image.jpg', path_overlay_img)
